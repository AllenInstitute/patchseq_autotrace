import re
import os
import cv2
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from tifffile import imwrite
from functools import partial
import allensdk.internal.core.lims_utilities as lu


def default_query_engine():
    """Get Postgres query engine with environmental variable parameters"""

    return partial(
        lu.query,
        host=os.getenv("LIMS_HOST"),
        port=5432,
        database=os.getenv("LIMS_DBNAME"),
        user=os.getenv("LIMS_USER"),
        password=os.getenv("LIMS_PASSWORD")
    )


def query_jp2_paths_and_indices(specimen_id, query_engine=None,static_paths_json=None):
    """
    Will return a list of dictionaries giving the path to jp2 files.
    """

    try:
        if query_engine is None:
            query_engine = default_query_engine()

        query = f"""
        SELECT  si.x AS x_ind, si.y AS y_ind , sl.storage_directory || im.jp2  AS input_jp2     
        FROM image_series iser     
        JOIN image_series_slides iss 
        ON iss.image_series_id=iser.id     
        JOIN slides sl ON sl.id=iss.slide_id     
        JOIN images im ON im.slide_id=sl.id 
        AND im.image_type_id = 1     
        JOIN sub_images si ON si.image_id=im.id     
        WHERE iser.specimen_id = {int(specimen_id)} 
        order by input_jp2 desc;
        """

        results = query_engine(query)

    except:
        results = []
        # static_paths_json = "/data/data/AWS_AutotraceTest_Paths.json"
        if static_paths_json is not None:
            with open(static_paths_json, "r") as f:
                paths_dict_list = json.load(f)['all_cells_jp2_paths']
            aibs_result = [sub_dict for sub_dict in paths_dict_list if sub_dict['specimen_id'] == int(specimen_id)][0]
            results = []
            for aibs_path in aibs_result['jp2_paths']:
                s3_path = aibs_path.replace("/allen/programs/celltypes/production/", "/data/data/")
                d = {}
                d['input_jp2'] = s3_path
                d['x_ind'] = 1
                d['y_ind'] = 1
                print(d)
                results.append(d)
        else:
            print("Could not complete query for jp2 paths and no static jp2 paths json provided. Nothing to return")


    return results

def get_63x_soma_coords(specimen_id, query_engine=None):
    """
    return the soma x and y coordinates as found in LIMS.

    :param specimen_id: int, specimen id
    :param query_engine: functools.partial
    :return: (array, array) x and y coordinates
    """

    try:
        if not query_engine:
            query_engine = default_query_engine()

        query = """
        select max(id) as image_series_id from image_series
        where specimen_id = {}
        group by specimen_id""".format(int(specimen_id))
        imser_id_63x = query_engine(query)[0]['image_series_id']

        sql_query = """
        select distinct 
                    cell.id as cell_id, 
                    ims63.id as image_series_63, 
                    layert.name as layer_type, 
                    si.specimen_tissue_index as z_index, 
                    poly.path as poly_path
        from specimens cell
        join image_series ims63 on ims63.specimen_id = cell.id
        join sub_images si on si.image_series_id = ims63.id
        join avg_graphic_objects layer on layer.sub_image_id = si.id
        join avg_group_labels layert on layert.id = layer.group_label_id
        join avg_graphic_objects poly on poly.parent_id = layer.id
        where ims63.id = {}
        """.format(imser_id_63x)

        res = query_engine(sql_query)
        soma_res = [d for d in res if d['layer_type'] == 'Soma']
        if len(soma_res) == 0:

            return None, None
        else:
            soma_res = soma_res[0]
            all_soma_coords = soma_res['poly_path']
            xs = list(map(int, all_soma_coords.split(",")[::2]))
            ys = list(map(int, all_soma_coords.split(",")[1::2]))
            return xs, ys
    except:
        print("Unabel to query lims for 63x soma coords, defaulting to naive soma location method")
        return None, None


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_tifs(input_tif_dir):
    tif_files = [file for file in os.listdir(input_tif_dir) if any([file.endswith(suf) for suf in ['.tif', 'tiff']])]
    return natural_sort(tif_files)


def dir_to_mip(indir, ofile, max_num_file_to_load, mip_axis=2):
    """
    Image stack directory to max intensity projection. Memory conscious. Will only load max_num_file_to_load at a time.
    dimensions : axis =  {x:1, y:0, z:2}

    in the case of yz or xz mips, we are 'building the mip as we go'. loading chunks along z, taking mip and saving.
    in case of xy mip, we are chunking, but keeping track of mips and then taking a final mip of mips.

    :param max_num_file_to_load: int, maximum number of tif files that will be loaded into memory at once
    :param indir: str, directory with tiff images in natural order
    :param ofile: str, output mip tif file
    :param mip_axis: int, axis for max intensity projection
    :return:

    """

    indir_files = get_tifs(indir)

    chunks = [indir_files[x:x + max_num_file_to_load] for x in range(0, len(indir_files), max_num_file_to_load)]

    img_0_pth = os.path.join(indir, indir_files[0])
    img_0 = cv2.imread(img_0_pth, cv2.IMREAD_UNCHANGED)
    if mip_axis == 0:
        final_mip = np.zeros((img_0.shape[1], len(indir_files)), dtype=np.uint8)
    elif mip_axis == 1:
        final_mip = np.zeros((img_0.shape[0], len(indir_files)), dtype=np.uint8)

    all_mips = []
    z_slice_ct = 0
    for sub_list in chunks:
        curr_stack = []
        for fn in sub_list:
            pth = os.path.join(indir, fn)
            img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
            curr_stack.append(img)
            z_slice_ct += 1

        curr_stack = np.dstack(curr_stack).astype(np.uint8)
        curr_mip = np.max(curr_stack, axis=mip_axis)

        idx_0 = z_slice_ct - len(sub_list)
        if mip_axis != 2:
            final_mip[:, idx_0:z_slice_ct] = curr_mip
        else:
            all_mips.append(curr_mip)

    # thing to consider: if datasets significantly grow in size,  len(all_mips) maybe > max_num_files_to_load
    if mip_axis == 2:
        all_mips = np.dstack(all_mips).astype(np.uint8)
        final_mip = np.max(all_mips, axis=mip_axis)

    final_mip = final_mip.astype(np.uint8)
    imwrite(ofile, final_mip)


def extract_non_zero_coords(tif_directory, thresh=None):
    """
    given a director of tif images, return a dataframe of all non-zero coordinates found in the image stack

    :param tif_directory: str/path, directory with tif images
    :param thresh: int below 255, threshold used to mask images. Values in the image below this will be replaced with 0
    :return:
    """

    these_tif_files = get_tifs(tif_directory)
    z_idx = -1
    records = []
    for fn in tqdm(these_tif_files):
        z_idx += 1
        pth = os.path.join(tif_directory, fn)
        img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
        if thresh:
            img[img < thresh] = 0
        ys, xs = np.nonzero(img)
        intensity = img[ys, xs]
        for x, y, i in zip(xs, ys, intensity):
            res_dict = {
                "x": x,
                "y": y,
                "z": z_idx,
                "Intensity": i
            }
            records.append(res_dict)

    non_zero_coords_df = pd.DataFrame.from_records(records)

    return non_zero_coords_df
