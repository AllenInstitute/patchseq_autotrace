import os
import argschema as ags
from patchseq_autotrace.processes.postprocess import postprocess


class IO_Schema(ags.ArgSchema):
    specimen_dir = ags.fields.InputDir(description='Input Subject Directory')
    model_name = ags.fields.Str(description='model name to use ')


def main(args, **kwargs):

    specimen_dir = args['specimen_dir']
    model_name = args['model_name']
    segmentation_dir = os.path.join(specimen_dir, "Segmentation")
    postprocess(specimen_dir, segmentation_dir=segmentation_dir, model_name=model_name)


def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
