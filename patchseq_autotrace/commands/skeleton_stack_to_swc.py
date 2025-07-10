import argschema as ags
from patchseq_autotrace.processes.stack_to_swc import skeleton_to_swc
from patchseq_autotrace.database_tools import status_update


class IO_Schema(ags.ArgSchema):
    specimen_dir = ags.fields.InputDir(description='Input Subject Directory')
    model_name = ags.fields.Str(description='model name to use ')
    sqlite_runs_table_id = ags.fields.Int(description="unique ID key for runs table in the sqlite .db file",allow_none=True,default=None)
    precalcualted_soma_x = ags.fields.Float(default=None,allow_nan=True, allow_none=True, description = '63x soma x location. Useful when running on a compute host without lims query')
    precalcualted_soma_y = ags.fields.Float(default=None,allow_nan=True, allow_none=True, description = '63x soma y location. Useful when running on a compute host without lims query')
    autotrace_tracking_database = ags.fields.Str(
        description="sqlite tracking .db file. This should exist and have specimen_runs table setup as seen in "
                    "patchseq_autotrace.database_tools prior to running this script",allow_none=True,default=None)


def main(args, **kwargs):
    specimen_dir = args['specimen_dir']
    model_name = args['model_name']
    precalcualted_soma_x = args['precalcualted_soma_x']
    precalcualted_soma_y = args['precalcualted_soma_y']
    sqlite_runs_table_id = args['sqlite_runs_table_id']
    autotrace_tracking_database = args['autotrace_tracking_database']
    if (sqlite_runs_table_id is not None) and (autotrace_tracking_database is not None ):

        status_update(database_path=autotrace_tracking_database,
                    runs_unique_id=sqlite_runs_table_id,
                    process_name='stack2swc',
                    state='start')

    skeleton_to_swc(specimen_dir=specimen_dir, 
                    model_and_version=model_name, 
                    precalcualted_soma_x=precalcualted_soma_x, 
                    precalcualted_soma_y = precalcualted_soma_y,
                    )

    if (sqlite_runs_table_id is not None) and (autotrace_tracking_database is not None ):
        
        status_update(database_path=autotrace_tracking_database,
                    runs_unique_id=sqlite_runs_table_id,
                    process_name='stack2swc',
                    state='finish')

def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
