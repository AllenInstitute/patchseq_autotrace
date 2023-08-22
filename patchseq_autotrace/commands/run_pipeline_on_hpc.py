import os
import argschema as ags
import pandas as pd
from patchseq_autotrace.slurm_tools.slurm_tools import submit_specimen_pipeline_to_slurm, \
    remove_already_autotrace_specimens
from patchseq_autotrace.database_tools import create_runs_table


class IO_Schema(ags.ArgSchema):

    specimen_file = ags.fields.InputFile(description='Input CSV to specimen ids')

    specimen_id_col = ags.fields.Str(default='Cell Specimen Id',
                                     description="Column to find specimen ids in input file")

    model_column = ags.fields.Str(default='model_to_use',
                                  description='column name depicting which model to segment with')

    chunk_size = ags.fields.Int(default=32,
                                description="Num Tif Images to Stack into Chunks")

    gpu_device = ags.fields.Int(default=0,
                                description="which gpu device to use for segmentation")

    virtual_environment = ags.fields.Str(description="Name of virtual environment")

    autotrace_root_directory = ags.fields.InputDir(default="/allen/programs/celltypes/workgroups/mousecelltypes"
                                                           "/AutotraceReconstruction")

    max_num_specimens_at_once = ags.fields.Int(description="maximum number of specimens to be running at once on hpc")

    autotrace_tracking_database = ags.fields.OutputFile(default="/allen/programs/celltypes/workgroups/mousecelltypes"
                                                           "/AutotraceReconstruction/Autotrace_DataBase.db")

def main(args, **kwargs):
    specimen_file = args['specimen_file']
    specimen_id_col = args['specimen_id_col']
    model_column = args['model_column']
    chunk_size = args['chunk_size']
    gpu_device = args['gpu_device']
    virtual_environment = args['virtual_environment']
    autotrace_root_directory = os.path.abspath(args['autotrace_root_directory'])
    max_n = args['max_num_specimens_at_once']
    autotrace_tracking_database = os.path.abspath(args['autotrace_tracking_database'])

    # Will create the runs table if it does not exist
    create_runs_table(autotrace_tracking_database)

    if not os.path.exists(specimen_file):
        raise ValueError("Specified input path does not exist")

    sps_df = pd.read_csv(specimen_file)

    if not all([c in sps_df.columns for c in [specimen_id_col, model_column]]):
        raise ValueError(f"Please make sure both {specimen_id_col} and {model_column} columns are in the input csv")

    if not os.path.exists(autotrace_root_directory):
        os.mkdir(autotrace_root_directory)

    # remove any cells that have already been autotraced
    sps_df = remove_already_autotrace_specimens(input_df=sps_df,
                                                specimen_id_col=specimen_id_col,
                                                autotrace_root_dir=autotrace_root_directory,
                                                model_name_col=model_column)

    if sps_df.empty:
        print("All Specimens Have Already Been Autotraced, Congrats")
        return None

    # Chunk the list of cells into smaller batches since we only allow max_n specimens to run at a given time
    df_indices = sps_df.index.tolist()
    chunked_indices = [df_indices[x:x + max_n] for x in range(0, len(df_indices), max_n)]

    # the first batch of cells do not have any dependencies
    parent_job_id_list = [None] * len(chunked_indices[0])
    parent_job_id_start_cond_list = [None] * len(chunked_indices[0])
    for idx_chunk in chunked_indices:

        these_sps = sps_df.loc[idx_chunk]

        cter = -1
        curr_parent_job_id_list = []
        curr_parent_job_id_start_cond_list = []
        for idx, row in these_sps.iterrows():
            # get specimens parent dependency (since we are limiting only max_n number of specimens to run at a time)
            cter += 1
            parent_job_id = parent_job_id_list[cter]
            start_condition = parent_job_id_start_cond_list[cter]
            sp_id = int(row[specimen_id_col])
            model_name = row[model_column]

            # create and submit specimen pipeline to slurm with dependencies
            specimens_last_job_id = submit_specimen_pipeline_to_slurm(specimen_id=sp_id,
                                                                      autotrace_directory=autotrace_root_directory,
                                                                      chunk_size=chunk_size,
                                                                      model_name=model_name,
                                                                      virtualenvironment=virtual_environment,
                                                                      parent_job_id=parent_job_id,
                                                                      start_condition=start_condition,
                                                                      gpu_device=gpu_device,
                                                                      database_file=autotrace_tracking_database)

            # Now cells from the subsequent batches will have to wait for an opening in a previous batch
            curr_parent_job_id_list.append(specimens_last_job_id)

            # current specimen doesn't care if parent job finishes successfully or not, just that it finishes.
            curr_parent_job_id_start_cond_list.append("afterany")

        parent_job_id_list = curr_parent_job_id_list
        parent_job_id_start_cond_list = curr_parent_job_id_start_cond_list


def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
