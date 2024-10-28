import os
import argschema as ags
from patchseq_autotrace.processes.postprocess import postprocess
from patchseq_autotrace.database_tools import status_update


class IO_Schema(ags.ArgSchema):
    specimen_dir = ags.fields.InputDir(description='Input Subject Directory')
    model_name = ags.fields.Str(description='model name to use ')
    sqlite_runs_table_id = ags.fields.Int(description="unique ID key for runs table in the sqlite .db file",default=None,allow_none=True)
    autotrace_tracking_database = ags.fields.InputFile(
        description="sqlite tracking .db file. This should exist and have specimen_runs table setup as seen in "
                    "patchseq_autotrace.database_tools prior to running this script",default=None,allow_none=True)


def main(args, **kwargs):

    specimen_dir = args['specimen_dir']
    model_name = args['model_name']
    sqlite_runs_table_id = args['sqlite_runs_table_id']
    autotrace_tracking_database = args['autotrace_tracking_database']
    if (sqlite_runs_table_id is not None) and (autotrace_tracking_database is not None):
        status_update(database_path=autotrace_tracking_database,
                    runs_unique_id=sqlite_runs_table_id,
                    process_name='postprocessing',
                    state='start')

    segmentation_dir = os.path.join(specimen_dir, "Segmentation")
    postprocess(specimen_dir, segmentation_dir=segmentation_dir, model_name=model_name)

    if (sqlite_runs_table_id is not None) and (autotrace_tracking_database is not None):
        status_update(database_path=autotrace_tracking_database,
                    runs_unique_id=sqlite_runs_table_id,
                    process_name='postprocessing',
                    state='finish')


def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
