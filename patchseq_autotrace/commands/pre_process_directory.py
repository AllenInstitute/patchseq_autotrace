import os
import argschema as ags
import json
from patchseq_autotrace.processes.pre_process import (crop_dimensions, crop_and_invert_directory_multiproc,
                                                      solve_for_bounding_box, convert_stack_to_3dchunks,
                                                      get_image_stack_for_specimen)
from patchseq_autotrace.utils import dir_to_mip
from patchseq_autotrace.database_tools import status_update


class IO_Schema(ags.ArgSchema):
    specimen_dir = ags.fields.InputDir(description='Input Subject Directory, expecting Single_Tif_Images subdir exists')
    chunk_size = ags.fields.Int(default=32, description="Num Tif Images to Stack into Chunks")
    sqlite_runs_table_id = ags.fields.Int(description="unique ID key for runs table in the sqlite .db file")
    autotrace_tracking_database = ags.fields.InputFile(
        description="sqlite tracking .db file. This should exist and have specimen_runs table setup as seen in "
                    "patchseq_autotrace.database_tools prior to running this script")


def main(args, **kwargs):
    specimen_dir = args['specimen_dir']
    chunk_size = args['chunk_size']
    sqlite_runs_table_id = args['sqlite_runs_table_id']
    autotrace_tracking_database = args['autotrace_tracking_database']

    status_update(database_path=autotrace_tracking_database,
                  runs_unique_id=sqlite_runs_table_id,
                  process_name='preprocessing',
                  state='start')

    specimen_id = os.path.basename(os.path.abspath(specimen_dir))
    print(specimen_id)
    # # Ensure image directory exists
    input_image_dir = os.path.join(specimen_dir, "Single_Tif_Images")
    if not os.path.exists(input_image_dir):
        get_image_stack_for_specimen(specimen_id, input_image_dir)

    elif len(os.listdir(input_image_dir)) == 0:
        get_image_stack_for_specimen(specimen_id, input_image_dir)

    # Find crop dimensions and crop the images
    x1, y1, x2, y2 = crop_dimensions(input_image_dir)
    crop_and_invert_directory_multiproc(input_image_dir, x1, x2, y1, y2, chunk_size)

    # Create bounding box file
    bb_file = os.path.join(specimen_dir, 'bbox_{}.json'.format(specimen_id))
    bound_box = solve_for_bounding_box(input_image_dir, chunk_size)
    bb_dict = {"specimen_id": specimen_id,
               "bounding_box": bound_box}
    with open(bb_file, "w") as f:
        json.dump(bb_dict, f)

    # Generate MIP (memory efficient)
    mip_ofile = os.path.join(specimen_dir, "Single_Tif_Images_Mip.tif")
    if not os.path.exists(mip_ofile):
        dir_to_mip(indir=input_image_dir, ofile=mip_ofile, max_num_file_to_load=chunk_size, mip_axis=2)

    # Convert directory with single tif files to 3d chunks for segmentation
    chunk_dir = os.path.join(specimen_dir, "Chunks_of_{}".format(chunk_size))
    if not os.path.exists(chunk_dir):
        os.mkdir(chunk_dir)
    convert_stack_to_3dchunks(chunk_size, input_image_dir, chunk_dir)

    status_update(database_path=autotrace_tracking_database,
                  runs_unique_id=sqlite_runs_table_id,
                  process_name='preprocessing',
                  state='finish')

def console_script():
    this_module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(this_module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
