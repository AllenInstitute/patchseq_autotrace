import os
import cv2
import numpy as np
from tqdm import tqdm
from tifffile import imsave
from patchseq_autotrace.utils import get_tifs, query_jp2_paths_and_indices
from patchseq_autotrace.Jp2ToTif import JP2ToTiff
from multiprocessing import Pool


def run_converter(cvrt, e):
    cvrt.run()


def get_image_stack_for_specimen(specimen_id, tif_output_directory, static_jp2_paths_file, downsample=1, parallel=True):
    """
    Will convert LIMS jp2ks to tif files

    :param specimen_id:
    :param tif_output_directory:
    :param downsample:
    :param parallel:
    :return:
    """

    if not os.path.exists(tif_output_directory):
        os.mkdir(tif_output_directory)

    arg_dicts = query_jp2_paths_and_indices(specimen_id, static_paths_json=static_jp2_paths_file)
    parallel_inputs = []
    for d in arg_dicts:

        new_fn = os.path.basename(d['input_jp2']).replace(".jp2", ".tif")
        tif_outfile = os.path.join(tif_output_directory, new_fn)

        d['downsample'] = downsample
        d['block_read'] = 2
        d['tiff_output_file'] = tif_outfile
        converter = JP2ToTiff(d)
        if parallel:
            parallel_inputs.append((converter, []))
        else:
            converter.run()

    if parallel:
        p = Pool()
        p.starmap(func=run_converter, iterable=parallel_inputs)


def crop_dimensions(input_tif_dir, base=64):
    """
    Will crop an image directory so that x and y dimensions are a multiple of 64

    :param input_tif_dir:
    :param base:
    :return:
    """
    # find crop dimensions
    filename_to_extract_crop_info = get_tifs(input_tif_dir)[0]
    uncropped_img = cv2.imread(os.path.join(input_tif_dir, filename_to_extract_crop_info), cv2.IMREAD_UNCHANGED)

    height, width = uncropped_img.shape

    # base = 64, we need our image stacks to have an x and y dimension multiple of 64
    height_nearest_mult_below = base * int(height / base)
    width_nearest_mult_below = base * int(width / base)

    x1, y1 = 0, 0
    x2, y2 = width_nearest_mult_below, height_nearest_mult_below

    return x1, y1, x2, y2


def _crop_and_invert(infile, ofile, x1, x2, y1, y2):
    """
    crop inpuyt file given inices and invert the greyscale color vals
    :param infile:
    :param ofile:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :return:
    """
    img = cv2.imread(infile, cv2.IMREAD_UNCHANGED)
    cropped_img = img[y1:y2, x1:x2]
    cropped_img_inverted = 255 - cropped_img
    cv2.imwrite(ofile, cropped_img_inverted)


def crop_and_invert_directory_multiproc(input_tif_dir, x1, x2, y1, y2, chunk_size):
    parallel_func_inputs = []
    tif_files = get_tifs(input_tif_dir)
    for fn in tif_files:
        infile = os.path.join(input_tif_dir, fn)
        ofile = infile
        parallel_func_inputs.append((infile, ofile, x1, x2, y1, y2))

    p = Pool(processes=chunk_size)
    p.starmap(_crop_and_invert, parallel_func_inputs)


def solve_for_bounding_box(input_image_dir, chunk_size):
    cropped_image = get_tifs(input_image_dir)[0]
    img_for_shape = cv2.imread(os.path.join(input_image_dir, cropped_image), cv2.IMREAD_UNCHANGED)
    inverted_height, inverted_width = img_for_shape.shape
    bound_box = [0, 0, 0, inverted_width, inverted_height, chunk_size]
    return bound_box


def convert_stack_to_3dchunks(chunk_size, single_tif_dir, chunk_dir):
    """
    will convert a directory full of single tif images into a directory with 3d chunks of images

    :param chunk_size: number of slices/individual tif files per chunk
    :param single_tif_dir: directory with individual slices/2d tifs
    :param chunk_dir: output directory
    :return: None, outut directory is populated
    """

    chunk_n = 0
    counter = 0
    cv_stack = []
    list_of_files = get_tifs(single_tif_dir)
    print("Generating 3D Chunks:")
    for files in tqdm(list_of_files):
        counter += 1
        img = cv2.imread(os.path.join(single_tif_dir, files), cv2.IMREAD_UNCHANGED)
        cv_stack.append(img)
        if counter == chunk_size:
            chunk_n += 1
            cv_stack = np.asarray(cv_stack).astype(np.uint8)

            imsave(os.path.join(chunk_dir, 'chunk{}.tif'.format(chunk_n)), cv_stack)
            cv_stack = []
            counter = 0
    if (float(len(list_of_files)) / float(chunk_size)).is_integer():
        print('{} files, {} chunk size'.format(len(list_of_files), chunk_size))
        print('How Lucky, the Last Chunk was a multiple of {}'.format(chunk_size))
    else:
        chunk_n += 1
        last_counter = 0
        last_cv_stack = []
        for files in list_of_files[-chunk_size:]:
            last_counter += 1
            last_img = cv2.imread(os.path.join(single_tif_dir, files), cv2.IMREAD_UNCHANGED)
            last_cv_stack.append(last_img)
            if last_counter == chunk_size:
                last_cv_stack = np.asarray(last_cv_stack).astype(np.uint8)
                imsave(os.path.join(chunk_dir, 'chunk{}.tif'.format(chunk_n)), last_cv_stack)
