import os
import json
import argschema as ags
from patchseq_autotrace.processes.validation import validate


class IO_Schema(ags.ArgSchema):
    # expecting that there is already in the specimen_dir:
    # /Single_Tif_Images
    # /Chunks_of_{chunk_size}
    # /Single_Tif_Images_Mip.tiff
    # /bbox_{specimen_id}.csv
    specimen_dir = ags.fields.InputDir(description='Input Subject Directory')
    chunk_size = ags.fields.Int(default=32, description="Num Tif Images to Stack into Chunks")
    model_name = ags.fields.Str(description='path to model checkpoint')
    gpu_device = ags.fields.Int(default=0)


def main(args, **kwargs):
    specimen_dir = args['specimen_dir']
    chunk_size = args['chunk_size']
    model_name = args['model_name']
    gpu_device = args['gpu_device']

    specimen_id = os.path.basename(os.path.abspath(specimen_dir))
    chunk_dir = os.path.join(specimen_dir, 'Chunks_of_{}'.format(chunk_size))
    bbox_path = os.path.join(specimen_dir, 'bbox_{}.json'.format(specimen_id))
    with open(bbox_path, "r") as f:
        bb = json.load(f)['bounding_box']
    # df = pd.read_csv(bbox_path)
    # bb = df.bound_boxing.values

    validate(model_name, specimen_dir, chunk_dir, bb, gpu_device, chunk_size)


def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
