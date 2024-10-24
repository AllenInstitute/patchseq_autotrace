import os
import shutil
import argschema as ags
from patchseq_autotrace.database_tools import status_update
from patchseq_autotrace.slurm_tools.Slurm_DAG import Slurm_DAG


class IO_Schema(ags.ArgSchema):
    specimen_dir = ags.fields.InputDir(description='Input Subject Directory')
    sqlite_runs_table_id = ags.fields.Int(description="unique ID key for runs table in the sqlite .db file")
    autotrace_tracking_database = ags.fields.InputFile(
        description="sqlite tracking .db file. This should exist and have specimen_runs table setup as seen in "
                    "patchseq_autotrace.database_tools prior to running this script")
    # the below are only if swc post processing will be run
    post_processing_workflow = ags.fields.Str(default = None, allow_none=True, description = "if not None, will submit postprocessing job here")
    job_dir = ags.fields.InputDir(default = "None",description= "Directory with job files for this cells run")
    model_name = ags.fields.Str(default = "None", description = "model name and version")
    
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
    
    # If swc post processing workflow was given as an input, submit a job from here. We dont want
    # this job tracked in our SLURM DAG because post processing takes a while to run as seen in time request
    job_dir = args['job_dir']
    specimen_id = os.path.basename(args['specimen_dir'])
    post_processing_workflow = args['post_processing_workflow']
    model_name = args['model_name']
    if post_processing_workflow is not None:
        # swc post processing
        swc_pp_job_file = os.path.join(job_dir, f"{specimen_id}_swc_post_proc.sh")
        swc_pp_kwargs = {
            "--job-name": f"swcPP-{specimen_id}",
            "--mail-type": "NONE",
            "--cpus-per-task": "4",
            "--nodes": "1",
            "--kill-on-invalid-dep": "yes",
            "--mem": "40gb",
            "--time": "85:00:00",
            "--partition": "celltypes",
            "--output": os.path.join(job_dir, f"{specimen_id}_swc_post_proc.log")
        }
        swc_pp_command = f'swc_postprocess --specimen_dir {specimen_dir} --post_processing_workflow "{post_processing_workflow}" --model_name "{model_name}"'
        swc_pp_list = ["source ~/.bashrc", "conda activate StableNeuronMorphNew", swc_pp_command]

        slurm_dag_node = {
            "id": 1,
            "parent_id": -1,
            "name": "swc_post_process",
            "slurm_kwargs": swc_pp_kwargs,
            "slurm_commands": swc_pp_list,
            "job_file": swc_pp_job_file
        }
    
        pipeline_dag = Slurm_DAG([slurm_dag_node])
        pipeline_dag.submit_dag_to_scheduler(parent_job_id=None,
                                            start_condition=None)
    

def console_script():
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=IO_Schema)
    main(module.args)
