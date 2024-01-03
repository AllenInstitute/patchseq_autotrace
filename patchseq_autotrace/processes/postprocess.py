import numpy as np
import tifffile as tif
import os
import cv2
from skimage.morphology import remove_small_objects, skeletonize_3d
from scipy import ndimage as ndi
import shutil
import pandas as pd
from patchseq_autotrace.utils import dir_to_mip, get_tifs, extract_non_zero_coords
from patchseq_autotrace.statics import INTENSITY_THRESHOLDS


def postprocess(specimen_dir, segmentation_dir, model_name, threshold=0.3, size_threshold=2000,
                max_stack_size=7000000000):
    """

    :param specimen_dir:
    :param segmentation_dir:
    :param model_name:
    :param threshold:
    :param size_threshold: minimum size a connected component must be
    :param max_stack_size:
    :return:
    """
    print("Starting TO Post-Process")
    intensity_threshold = INTENSITY_THRESHOLDS[model_name]

    savedir = os.path.join(specimen_dir, 'Skeleton')
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    soma_intensity_thresh = intensity_threshold
    axon_dendrite_intensity_thresh = int(np.round(255 * threshold))
    threshold_dict = {
        "ch1": soma_intensity_thresh,
        "ch2": axon_dendrite_intensity_thresh,
        "ch3": axon_dendrite_intensity_thresh,
    }
    print("Starting TO Make Segmentation CSVs")
    for chan, thresh in threshold_dict.items():
        print(chan)
        channel_dir = os.path.join(segmentation_dir, chan)
        xy_mip = os.path.join(specimen_dir, 'MAX_Segmentation_{}.tif'.format(chan))
        dir_to_mip(indir=channel_dir, ofile=xy_mip, max_num_file_to_load=32, mip_axis=0)

        if chan == 'ch1':
            yz_mip_ofile = os.path.join(specimen_dir, 'MAX_yz_Segmentation_{}.tif'.format(chan))
            dir_to_mip(indir=channel_dir, ofile=yz_mip_ofile, max_num_file_to_load=32, mip_axis=2)

        # find x,y,z and intensity values for non zero coordinates
        csv_ofile = os.path.join(specimen_dir, "Segmentation_{}.csv".format(chan))
        cgx, cgy, cgz = extract_non_zero_coords(tif_directory=channel_dir, thresh=thresh, max_list_size=500000,
                                                output_csv=csv_ofile)

        if (chan == "ch1") and (not all([c == 0 for c in [cgx, cgy, cgz]])):
            centroid_ofile = os.path.join(specimen_dir, "Segmentation_soma_centroid.csv")
            np.savetxt(centroid_ofile, np.swapaxes(np.array([[cgx], [cgy], [cgz]]), 0, 1), fmt='%.1f', delimiter=',',
                       header='x,y,z')

    # create folder for arbor segmentation
    ch5_dir = os.path.join(segmentation_dir, 'ch5')
    if not os.path.isdir(ch5_dir):
        os.mkdir(ch5_dir)

    mask_dir = os.path.join(segmentation_dir, 'mask')
    if not os.path.isdir(mask_dir):
        os.mkdir(mask_dir)

    ch2_dir = os.path.join(segmentation_dir, "ch2")
    ch3_dir = os.path.join(segmentation_dir, "ch3")

    filelist = get_tifs(ch2_dir)
    for i, f in enumerate(filelist):
        filename = os.path.join(ch2_dir, f)
        img2 = tif.imread(filename)
        filename = os.path.join(ch3_dir, f)
        img3 = tif.imread(filename)
        img = np.maximum(img2, img3)
        tif.imsave(os.path.join(ch5_dir, '%03d.tif' % i), img)  # consecutive numbers
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[img2 > img3] = 2
        mask[img3 > img2] = 3
        tif.imsave(os.path.join(mask_dir, '%03d.tif' % i), mask)  # consecutive numbers

    # Step 5. Postprocess arbor segmentation
    filelist = get_tifs(ch5_dir)

    cell_stack_size = (len(filelist), img.shape[0], img.shape[1])
    cell_stack_memory = cell_stack_size[0] * cell_stack_size[1] * cell_stack_size[2]
    print('cell_stack_size (z,y,x):', cell_stack_size, cell_stack_memory)
    # if cell stack memory>max_stack_size (15GB for RAM=128GB), need to split
    num_parts = int(np.ceil(cell_stack_memory / max_stack_size))
    print('num_parts:', num_parts)

    # split filelist
    idx = np.append(np.arange(0, cell_stack_size[0], int(np.ceil(cell_stack_size[0] / num_parts))),
                    cell_stack_size[0] + 1)
    print('idx:', idx)
    for i in range(num_parts):
        idx1 = idx[i]
        idx2 = idx[i + 1]
        filesublist = filelist[idx1:idx2]
        print('part ', i, idx1, idx2, len(filesublist))

        # load stack
        stack_size = len(filesublist), cell_stack_size[1], cell_stack_size[2]
        stack = np.zeros(stack_size, dtype=np.uint8)
        print('loading stack')
        for j, f in enumerate(filesublist):
            filename = os.path.join(ch5_dir, f)
            img = tif.imread(filename)
            stack[j, :, :] = img
        print(stack.shape, stack.dtype)

        print('removing smaller segments')
        # binarize stack based on threshold
        # stack = (stack > int(np.round(255 * threshold))).astype(np.uint8)
        stack = (stack > axon_dendrite_intensity_thresh).astype(np.uint8)

        # label connected components
        s = ndi.generate_binary_structure(3, 3)
        stack = ndi.label(stack, structure=s)[0].astype(np.uint16)

        # remove components smaller than size_threshold
        # connectivity=3 - pixels are connected if their faces, edges, or corners touch
        stack = remove_small_objects(stack, min_size=size_threshold, connectivity=3)

        # convert all connected component labels to 1
        stack = (stack > 0).astype(np.uint8)

        # skeletonize stack
        print('skeletonizing stack')
        stack = skeletonize_3d(stack)

        # save stack as multiple tif files
        print('saving stack')
        for k in range(stack.shape[0]):
            tif.imsave(os.path.join(savedir, '%03d.tif' % (k + idx1)), stack[k, :, :])

    # save skeleton as csv file
    skeleton_coords_csv = os.path.join(specimen_dir, "Segmentation_skeleton_labeled.csv")
    _, _, _ = extract_non_zero_coords(tif_directory=savedir,
                                      output_csv=skeleton_coords_csv,
                                      max_list_size=500000,
                                      thresh=None)
    # this ~should~ be small enough to read into memory
    skeleton_df = pd.read_csv(skeleton_coords_csv)

    skeleton_df['node_type'] = [None] * len(skeleton_df)
    z_slices_with_segmentation = set(skeleton_df['z'].values)

    mask_tif_files = get_tifs(mask_dir)
    z_idx = -1
    for fn in mask_tif_files:
        z_idx += 1

        if z_idx in z_slices_with_segmentation:
            pth = os.path.join(mask_dir, fn)
            msk_img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
            nodes_at_this_slice = skeleton_df[skeleton_df['z'] == z_idx]
            xs = nodes_at_this_slice['x'].values
            ys = nodes_at_this_slice['y'].values
            axon_dendrite_label = msk_img[ys, xs]
            axon_dendrite_label[axon_dendrite_label == 0] = 5

            skeleton_df.loc[skeleton_df['z'] == z_idx, 'node_type'] = axon_dendrite_label

    skeleton_df.to_csv(skeleton_coords_csv, index=False)

    directories_to_remove = ["Chunks_of_32", "Chunks_of_32_Left", "Chunks_of_32_Right",
                             "Segmentation", "Segmentation_Left", "Segmentation_Right",
                             "Single_Tif_Images", "Single_Tif_Images_Left", "Single_Tif_Images_Right"]

    for dir_name in directories_to_remove:
        full_dir_name = os.path.join(specimen_dir, dir_name)
        if os.path.exists(full_dir_name):
            print(full_dir_name)
            shutil.rmtree(full_dir_name)
