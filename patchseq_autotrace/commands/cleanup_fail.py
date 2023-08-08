import os
import shutil
import argschema as ags
from patchseq_autotrace.database_tools import status_update


class IO_Schema(ags.ArgSchema):
    specimen_dir = ags.fields.InputDir(description='Input Subject Directory')
    sqlite_runs_table_id = ags.fields.Int(description="unique ID key for runs table in the sqlite .db file")
    autotrace_tracking_database = ags.fields.InputFile(
        description="sqlite tracking .db file. This should exist and have specimen_runs table setup as seen in "
                    "patchseq_autotrace.database_tools prior to running this script")


def main(args, **kwargs):

    specimen_dir = args['specimen_dir']
    directories_to_remove = ["Chunks_of_32", "Chunks_of_32_Left", "Chunks_of_32_Right",
                             "Segmentation", "Left_Segmentation", "Right_Segmentation",
                             "Skeleton", "Left_Skeleton", "Right_Skeleton",
                             "Single_Tif_Images", "Single_Tif_Images_Left", "Single_Tif_Images_Right"]
    print("Cleaning Up:")
    for dir_name in directories_to_remove:
        full_dir_name = os.path.join(specimen_dir, dir_name)
        if os.path.exists(full_dir_name):
            print(full_dir_name)
            shutil.rmtree(full_dir_name)

    sqlite_runs_table_id = args['sqlite_runs_table_id']
    autotrace_tracking_database = args['autotrace_tracking_database']

    status_update(database_path=autotrace_tracking_database,
                  runs_unique_id=sqlite_runs_table_id,
                  process_name='pipeline',
                  state='finish')

def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
