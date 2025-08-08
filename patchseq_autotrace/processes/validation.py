import os
import shutil
import json
import tifffile as tif
from neuroseg.nets.RSUNetMulti import RSUNetMulti
from neuroseg.core.predictor_multilabel import Predictor
from neuroseg.datasets.datatypes import (BoundingBox, Vector)
from neuroseg.datasets.filetypes import TiffVolume
from neuroseg.datasets.dataset import Array
import numpy as np
from patchseq_autotrace.statics import MODEL_NAME_PATHS
from patchseq_autotrace.utils import natural_sort, get_tifs, get_jp2_slice_size_and_stack_length

def check_if_segmentation_exists(specimen_dir, chunk_dir, number_of_individual_tiffs):
    
    # Check if segmentation already completed
    seg_dir = os.path.join(specimen_dir, 'Segmentation')
    segmentation_complete = True
    for ch in range(1, 4):
        ch_dir = os.path.join(seg_dir, f"ch{ch}")
        if not os.path.isdir(ch_dir):
            segmentation_complete = False
            break
        tiff_files = get_tifs(ch_dir)
        if len(tiff_files) != number_of_individual_tiffs:
            segmentation_complete = False
            break

    if segmentation_complete:
        print(f"Segmentation already completed for: \n{specimen_dir}")
        if os.path.isdir(chunk_dir):
            print("Still cleaning up existing chunk_dir.")
            shutil.rmtree(chunk_dir)
        return segmentation_complete
    
    return segmentation_complete
    
def validate(model_name_version, specimen_dir, chunk_dir, bb, gpu, chunk_size):
    seg_dir = os.path.join(specimen_dir, 'Segmentation')
    if not os.path.isdir(seg_dir):
        os.mkdir(seg_dir)
    
    specimen_id = os.path.basename(os.path.abspath(specimen_dir))
    # to prevent unnecessary query, just load from the json
    # _, number_of_individual_tiffs = get_jp2_slice_size_and_stack_length(specimen_id)
    # if number_of_individual_tiffs is None:
    bb_file = os.path.join(specimen_dir, 'bbox_{}.json'.format(specimen_id))
    
    with open(bb_file, "r") as f:
        bound_box_dict = json.load(f)
    
    number_of_individual_tiffs = bound_box_dict['num_tiff_files']
    
    segmentation_complete = check_if_segmentation_exists(specimen_dir, chunk_dir, number_of_individual_tiffs)
    print(f"From within the segmentation script, segmentation_complete bool = {segmentation_complete}")
    if segmentation_complete:
        return 
    
    net = RSUNetMulti()
    data_text = MODEL_NAME_PATHS[model_name_version]

    count = [0, 0, 0]
    number_of_small_segments = len(get_tifs(chunk_dir))
    print('I think there are {} chunk tiff files in:\n {}'.format(number_of_small_segments, chunk_dir))
    for n in range(1, number_of_small_segments + 1):
        bbn = BoundingBox(Vector(bb[0], bb[1], bb[2]), Vector(bb[3], bb[4], chunk_size))

        nth_tiff_stack = os.path.join(chunk_dir, 'chunk{}.tif'.format(n))
        with TiffVolume(nth_tiff_stack, bbn) as inputs:

            # Predict
            predictor = Predictor(net, data_text, gpu_device=gpu)
            # output_volume is a list (len3) of Arrays for each of 3 foreground channels (soma, axon, dendrite)
            output_volume = [Array(np.zeros(inputs.getBoundingBox().getNumpyDim(), dtype=np.uint8)) for _ in range(3)]
            print('bb0', inputs.getBoundingBox())
            predictor.run(inputs, output_volume)

            for ch in range(3):
                ch_dir = os.path.join(seg_dir,"ch{}".format(ch+1))
                #ch_dir = os.path.join(seg_dir, 'ch%d ' % (ch + 1))
                if not os.path.isdir(ch_dir):
                    os.mkdir(ch_dir)
                probability_map = output_volume[ch].getArray()
                for i in range(probability_map.shape[0]):  # save as multiple tif files
                    # print('Prob Map Shape= ', probability_map.shape[0])
                    count[ch] += 1
                    this_image = probability_map[i, :, :]
                    this_image = this_image.astype(np.uint8)

                    im_file = os.path.join(ch_dir, '%03d.tif' % (count[ch]))
                    tif.imsave(im_file, this_image)

        
    for ch in range(3):
        # ch_dir = os.path.join(seg_dir, 'ch%d' % (ch + 1))
        ch_dir = os.path.join(seg_dir, "ch{}".format(ch + 1))
        number_of_segmented_tiffs = len([f for f in os.listdir(ch_dir) if '.tif' in f])
        print('Number of individual tiffs = {}'.format(number_of_individual_tiffs))
        print('Number of segmented tiffs = {}'.format(number_of_segmented_tiffs))

        number_of_duplicates = number_of_segmented_tiffs - number_of_individual_tiffs
        # assigning the number of duplicates to the difference in length between segmented dir and individual
        # tiff dir.
        if number_of_duplicates == 0:
            print('no duplicates were made')
            print('num duplicates = {}'.format(number_of_duplicates))

        else:
            print('num duplicates = {}'.format(number_of_duplicates))
            # this means that list_of_segmented_files[-32:-number_of_uplicates] can be erased because of preprocessing
            list_of_segmented_files = [x for x in natural_sort(get_tifs(ch_dir))]
            second_index = chunk_size - number_of_duplicates
            duplicate_segmentations = list_of_segmented_files[-chunk_size:-(second_index)]
            print(duplicate_segmentations)

            for files in duplicate_segmentations:
                os.remove(os.path.join(ch_dir, files))

    # delete chunk dir as soon as we no longer need it
    print("Deleting Chunked Tif Directory")
    shutil.rmtree(chunk_dir)