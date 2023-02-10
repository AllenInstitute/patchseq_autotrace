import os
import shutil
import argschema as ags


class IO_Schema(ags.ArgSchema):
    specimen_dir = ags.fields.InputDir(description='Input Subject Directory')


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

def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
