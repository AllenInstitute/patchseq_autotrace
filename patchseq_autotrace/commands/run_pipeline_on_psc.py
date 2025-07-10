import os
import argschema as ags
import pandas as pd
from patchseq_autotrace.slurm_tools.slurm_tools import submit_specimen_pipeline_to_slurm, \
    remove_already_autotrace_specimens
from patchseq_autotrace.database_tools import create_runs_table


class IO_Schema(ags.ArgSchema):

    specimen_file = ags.fields.InputFile(description='Input CSV to specimen ids')

    specimen_id_col = ags.fields.Str(default='Cell Specimen Id',
                                     description="This column should exist in the specimen_file input file")

    model_column = ags.fields.Str(default='model_to_use',
                                  description='column name depicting which model to segment with. Options are ["Aspiny1.0", "Spiny1.0"] for gluta and gaba data respectively ')

    soma_x_column = ags.fields.Str(default='average_soma_x',
                                  description='column name depicting mean 63x soma location in x dimension '
                                  )
    soma_y_column = ags.fields.Str(default='average_soma_y',
                                  description='column name depicting mean 63x soma location in y dimension '
                                  )
    image_storage_location_column = ags.fields.Str(default='bildirectory_image_stack',
                                  description='column name depicting image stack location (only applicable for data already published and hosted on BIL) ',
                                  allow_none=True)

    chunk_size = ags.fields.Int(default=32,
                                description="Num Tif Images to Stack into Chunks")

    gpu_device = ags.fields.Int(default=0,
                                description="which gpu device to use for segmentation")

    virtual_environment = ags.fields.Str(description="Name of virtual environment SLURM jobs will activate to run. patchseq_autotrace must be installed in this environemnt")

    autotrace_root_directory = ags.fields.InputDir(default="/bil/proj/um1lein/autotrace/AccessAutotraceReconstruction",
                                                           description="root directory where all processing will occur"  
                                                           )

    max_num_specimens_at_once = ags.fields.Int(description="maximum number of specimens to be running at once on hpc")

    dynamic_resource_requests = ags.fields.Bool(description='whether to change HPC resource requests depending on estimated image stack size')

    autotrace_tracking_database = ags.fields.Str(default="/bil/proj/um1lein/autotrace/AccessAutotraceReconstruction/Autotrace_DataBase.db", allow_none=True)
    
    post_processing_workflow_column = ags.fields.Str(default=None,
                                              description = "column name in specimen_file depicting which post-processing workflow to run",
                                              allow_none=True) 
    
    anaconda_version_to_activate = ags.fields.Str(default="anaconda3/2024.10-1", description='which anaconda module to load on bridges compute cluster')
    
def main(args, **kwargs):
    dynamic_resource_requests = args['dynamic_resource_requests']
    specimen_file = args['specimen_file']
    specimen_id_col = args['specimen_id_col']
    model_column = args['model_column']
    chunk_size = args['chunk_size']
    gpu_device = args['gpu_device']
    virtual_environment = args['virtual_environment']
    autotrace_root_directory = os.path.abspath(args['autotrace_root_directory'])
    max_n = args['max_num_specimens_at_once']
    autotrace_tracking_database = args['autotrace_tracking_database'] 
    post_processing_workflow_column = args['post_processing_workflow_column']
    soma_x_column = args['soma_x_column']
    soma_y_column = args['soma_y_column']
    image_storage_location_column = args['image_storage_location_column']
    anaconda_version_to_activate = args['anaconda_version_to_activate']
        
    # Will create the runs table if it does not exist
    if autotrace_tracking_database == "None":
        autotrace_tracking_database = None
    if autotrace_tracking_database is not None:
        autotrace_tracking_database = os.path.abspath(autotrace_tracking_database)
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

            pp_workflow = None
            if post_processing_workflow_column is not None:
                pp_workflow = row[post_processing_workflow_column]
                
            input_image_storage_dir = row[image_storage_location_column]
            if input_image_storage_dir is None:
                
                specimen_dir = os.path.join(autotrace_root_directory, str(sp_id))
                input_image_storage_dir = os.path.join(specimen_dir, "Single_Tif_Images")
            else:
                print(f"For specimen:{sp_id}\nCSV Told me to get images from:\n{input_image_storage_dir}")
                
            bil_data_package = {
                'soma_x':row[soma_x_column],
                'soma_y':row[soma_y_column],
                'image_storage_location':input_image_storage_dir,    
            }
               
            # create and submit specimen pipeline to slurm with dependencies
            specimens_last_job_id = submit_specimen_pipeline_to_slurm(specimen_id=sp_id,
                                                                      autotrace_directory=autotrace_root_directory,
                                                                      chunk_size=chunk_size,
                                                                      model_name=model_name,
                                                                      virtualenvironment=virtual_environment,
                                                                      parent_job_id=parent_job_id,
                                                                      start_condition=start_condition,
                                                                      gpu_device=gpu_device,
                                                                      database_file=autotrace_tracking_database,
                                                                      dynamic_resource_requests=dynamic_resource_requests,
                                                                      post_processing_workflow=pp_workflow,
                                                                      bil_data_package = bil_data_package,
                                                                      anaconda_version_to_activate =anaconda_version_to_activate,
                                                                      )

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
