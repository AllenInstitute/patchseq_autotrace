import re
import os
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from tifffile import imwrite, imread
from functools import partial
import glob
import dask
import dask.array as da


def _connect(user, host, database,
             password, port):
    import pg8000

    conn = pg8000.connect(user=user, host=host, database=database,
                          password=password, port=port)
    return conn, conn.cursor()


def _select(cursor, query):
    cursor.execute(query)
    columns = [d[0].decode("utf-8") if isinstance(d[0], bytes) else d[0] for d
               in cursor.description]
    return [dict(zip(columns, c)) for c in cursor.fetchall()]


def query(query, user, host, database,
          password, port):
    conn, cursor = _connect(user, host, database, password, port)

    # Guard against non-ascii characters in query
    query = ''.join([i if ord(i) < 128 else ' ' for i in query])

    try:
        results = _select(cursor, query)
    finally:
        cursor.close()
        conn.close()
    return results

def default_query_engine():
    """Get Postgres query engine with environmental variable parameters"""

    return partial(
        query,
        host=os.getenv("LIMS_HOST"),
        port=5432,
        database=os.getenv("LIMS_DBNAME"),
        user=os.getenv("LIMS_USER"),
        password=os.getenv("LIMS_PASSWORD")
    )


def query_jp2_paths_and_indices(specimen_id, query_engine=None):
    """Get an SWC file path for a specimen ID using the specified query engine"""
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

    return results


def get_63x_soma_coords(specimen_id, query_engine=None):
    """
    return the soma x and y coordinates as found in LIMS.

    :param specimen_id: int, specimen id
    :param query_engine: functools.partial
    :return: (array, array) x and y coordinates
    """
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


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_tifs(input_tif_dir):
    tif_files = [file for file in os.listdir(input_tif_dir) if any([file.endswith(suf) for suf in ['.tif', 'tiff']])]
    return natural_sort(tif_files)


def dir_to_mip(indir, max_num_file_to_load, axis=0, ofile=None):
    """
    Calculates the max intensity projection of all tiff images in a directory
    Using a memory efficient dask array

    :param indir: str, The directory containing the tiff images.
    :param max_num_file_to_load: int, The number of slices (images) to load into memory at once.
    :param axis: int, The axis along which to perform max intensity projection. 0 is into slice to get x,y MIP
    :param ofile: str, Path to an output tif file, optional, default=None
    :return:mip: np.array, max intensity projection

    """
    tiff_files = sorted(glob.glob(os.path.join(indir, '*.tiff')))

    if not tiff_files:
        tiff_files = sorted(glob.glob(os.path.join(indir, '*.tif')))
        if not tiff_files:
            raise ValueError(f"No tiff files found in directory: {indir}")

    # Get the shape of the first image to use for subsequent images
    first_img_shape = imread(tiff_files[0]).shape

    # Lazy function to load a single tiff file
    def load_single_tiff(file):
        return imread(file)

    # Create a list of Dask delayed objects, each representing a lazy-loaded image
    lazy_images = [da.from_delayed(dask.delayed(load_single_tiff)(f), shape=first_img_shape, dtype=np.uint8) for f in
                   tiff_files]

    # Stack these delayed images into a dask array with the specified chunk size (`max_num_file_to_load`).
    stacked_images = da.stack(lazy_images, axis=0)
    stacked_images = stacked_images.rechunk((max_num_file_to_load, *first_img_shape))

    # Compute the max intensity projection along the specified axis
    mip = stacked_images.max(axis=axis).compute()

    if axis in [1, 2]:
        # For consistency with other autotrace infrastructure
        mip = mip.T

    if ofile is not None:
        imwrite(ofile, mip)

    return mip


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
