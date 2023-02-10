import argschema as ags
from patchseq_autotrace.processes.stack_to_swc import skeleton_to_swc


class IO_Schema(ags.ArgSchema):
    specimen_dir = ags.fields.InputDir(description='Input Subject Directory')
    model_name = ags.fields.Str(description='model name to use ')


def main(args, **kwargs):
    specimen_dir = args['specimen_dir']
    model_name = args['model_name']

    skeleton_to_swc(specimen_dir=specimen_dir, model_and_version=model_name)


def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
