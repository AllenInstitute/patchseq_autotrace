import os
from datetime import date
from patchseq_autotrace.slurm_tools.Slurm_DAG import Slurm_DAG
from patchseq_autotrace import __version__ as autotrace_code_version


def remove_already_autotrace_specimens(input_df, specimen_id_col, autotrace_root_dir, model_name_col):
    """
    Will remove already autotraced specimens from the current que dataframe

    :param input_df: (DataFrame): input dataframe
    :param specimen_id_col: (str): column indicating specimen id
    :param autotrace_root_dir: (str): autotrace root directory to search through
    :param model_name_col: (str): column indicating model to use
    :return input_df: (DataFrame) modified input dataframe with already autotraced cells removed
    """
    # check the specimen ids to see if they have been autotraced yet
    remove_from_this_run = []
    for idx, row in input_df.iterrows():
        specimen_id = str(int(row[specimen_id_col]))
        if specimen_id is not None:
            model_name = row[model_name_col]
            spdir = os.path.join(autotrace_root_dir, specimen_id)
            newest_raw_name = os.path.join(spdir, "SWC", "Raw", "{}_{}_{}_1.0.swc".format(specimen_id, model_name, autotrace_code_version))

            if os.path.exists(newest_raw_name):
                remove_from_this_run.append(specimen_id)

    input_df = input_df[~input_df[specimen_id_col].isin(remove_from_this_run)]

    return input_df


def submit_specimen_pipeline_to_slurm(specimen_id, autotrace_directory, chunk_size, model_name, virtualenvironment,
                                      parent_job_id, start_condition, gpu_device, gpu_batch_size, static_jp2_paths_file):
    """
    Will create a slurm workflow DAG for each step in the autotrace pipeline for the given specimen and submit the
    jobs to the slurm scheduler. Each step of the pipeline requires that the previous step be completed without fail
    (i.e. slurm dependency afterok). The kill-on-invalid-dep keyword will cause a job to be killed if the dependency
    will never be met (e.g. the parent job failed but the afterok keyword was supplied).

    TODO dynamically request resources depending on size of specimen image stack. Also considering disc quota

    :param specimen_id: (int): specimen id
    :param autotrace_directory: (str): path to root (all specimens) autotrace directory
    :param chunk_size: (int): number of z-slices in segmentation bounding box
    :param model_name: (str): model name, used to find model checkpoint files and hardcoded thresholds
    :param virtualenvironment: (str): name of virtual environment to be activated on HPC
    :param parent_job_id: (int): the job id that this specimen must wait to finish before it is allowed to begin
    :param start_condition: (str): slurm depednency conditions (afterok, afterany, etc.)
    :param gpu_device: (int): which gpu device to use for segmentation
    :param static_jp2_paths_file: (str): path to workaround json for getting jp2 paths on aws
    :return:
    """
    specimen_dir = os.path.abspath(os.path.join(autotrace_directory, str(specimen_id)))
    if not os.path.exists(specimen_dir):
        os.mkdir(specimen_dir)

    specimen_log_dir = os.path.join(specimen_dir, "HPC_Logs")
    if not os.path.exists(specimen_log_dir):
        os.mkdir(specimen_log_dir)

    today = date.today()
    todays_date = today.strftime("%b_%d_%Y")
    job_dir = os.path.join(specimen_log_dir, todays_date + "_0")
    stop_cond = False
    ct = 0
    while not stop_cond:
        ct += 1
        if not os.path.exists(job_dir):
            os.mkdir(job_dir)
            stop_cond = True
        else:
            job_dir = "_".join(job_dir.split("_")[0:-1])
            job_dir = job_dir + "_{}".format(ct)

    # configure slurm resources and commands for each step of processes

    # Pre Process
    pre_proc_job_file = os.path.join(job_dir, f"{specimen_id}_pre_proc.sh")
    pre_process_slurm_kwargs = {
        "--job-name": f"pre-proc-{specimen_id}",
        "--mail-type": "NONE",
        "--cpus-per-task": f"{chunk_size}",
        "--nodes": "1",
        "--kill-on-invalid-dep": "yes",
        "--mem": "80gb",
        "--time": "2:00:00",
        "--partition": "celltypes",
        "--output": os.path.join(job_dir, f"{specimen_id}_pre_proc.log")
    }
    pre_proc_command = f"auto-pre-proc --specimen_dir {specimen_dir} --chunk_size {chunk_size} --static_jp2_paths_file {static_jp2_paths_file}"
    if static_jp2_paths_file is None:
        pre_proc_command = pre_proc_command.replace(f"--static_jp2_paths_file {static_jp2_paths_file}" , "")

    pre_proc_command_list = ["source ~/.bashrc", f"conda activate {virtualenvironment}", pre_proc_command]

    # Segmentation
    segmentation_job_file = os.path.join(job_dir, f"{specimen_id}_segmentation.sh")
    segmentation_slurm_kwargs = {
        "--job-name": f"seg-{specimen_id}",
        "--mail-type": "NONE",
        "--nodes": "1",
        "--kill-on-invalid-dep": "yes",
        "--cpus-per-task": "8",
        "--mem": "62gb",
        "--time": "72:00:00",
        "--gpus": "v100:1",
        "--partition": "celltypesgpu", #change this to celltypesgpu
        "--output": os.path.join(job_dir, f"{specimen_id}_segmentation.log")
    }
    seg_command = f"auto-segmentation --specimen_dir {specimen_dir} --chunk_size {chunk_size} --model_name {model_name} --gpu_device {gpu_device} --gpu_batch_size {gpu_batch_size}"
    seg_command_list = ["source ~/.bashrc", f"conda activate {virtualenvironment}", seg_command]

    # Post-Process Segmentation
    post_proc_job_file = os.path.join(job_dir, f"{specimen_id}_post_proc.sh")
    post_proc_segmentation_slurm_kwargs = {
        "--job-name": f"post-proc-{specimen_id}",
        "--mail-type": "NONE",
        "--cpus-per-task": "1",
        "--nodes": "1",
        "--kill-on-invalid-dep": "yes",
        "--mem": "120gb",
        "--time": "48:00:00",
        "--partition": "celltypes",
        "--output": os.path.join(job_dir, f"{specimen_id}_post_proc.log")

    }
    post_proc_command = f"auto-post-proc --specimen_dir {specimen_dir} --model_name {model_name} "
    post_proc_command_list = ["source ~/.bashrc", f"conda activate {virtualenvironment}", post_proc_command]

    # Convert Post Processed Skeleton Stack To SWC
    skeleton_2_swc_job_file = os.path.join(job_dir, f"{specimen_id}_stack_2_swc.sh")
    skeleton_to_swc_slurm_kwargs = {
        "--job-name": f"stack2swc-{specimen_id}",
        "--mail-type": "NONE",
        "--cpus-per-task": "1",
        "--nodes": "1",
        "--kill-on-invalid-dep": "yes",
        "--mem": "120gb",
        "--time": "48:00:00",
        "--partition": "celltypes",
        "--output": os.path.join(job_dir, f"{specimen_id}_stack_2_swc.log")
    }
    stack_2_swc_command = f"auto-skeleton-to-swc --specimen_dir {specimen_dir} --model_name {model_name} "
    skeleton_2_swc_command_list = ["source ~/.bashrc", f"conda activate {virtualenvironment}", stack_2_swc_command]

    # Cleanup
    cleanup_job_file = os.path.join(job_dir, f"{specimen_id}_cleanup.sh")
    cleanup_kwargs = {
        "--job-name": f"cleanup-{specimen_id}",
        "--mail-type": "NONE",
        "--cpus-per-task": "8",
        "--nodes": "1",
        "--kill-on-invalid-dep": "yes",
        "--mem": "4gb",
        "--time": "2:00:00",
        "--partition": "celltypes",
        "--output": os.path.join(job_dir, f"{specimen_id}_cleanup.log")
    }
    cleanup_command = f"auto-cleanup --specimen_dir {specimen_dir} "
    cleanup_command_list = ["source ~/.bashrc", f"conda activate {virtualenvironment}", cleanup_command]

    # Build the node list needed to construct a workflow dag
    slurm_dag_node_list = [

        {
            "id": 1,
            "parent_id": -1,
            "name": "pre-process",
            "slurm_kwargs": pre_process_slurm_kwargs,
            "slurm_commands": pre_proc_command_list,
            "job_file": pre_proc_job_file
        },

        {
            "id": 2,
            "parent_id": 1,
            "name": "segmentation",
            "slurm_kwargs": segmentation_slurm_kwargs,
            "slurm_commands": seg_command_list,
            "job_file": segmentation_job_file,
            "start_condition": 'afterok',
        },

        {
            "id": 3,
            "parent_id": 2,
            "name": "post-proccess",
            "slurm_kwargs": post_proc_segmentation_slurm_kwargs,
            "slurm_commands": post_proc_command_list,
            "job_file": post_proc_job_file,
            "start_condition": 'afterok',
        },

        {
            "id": 4,
            "parent_id": 3,
            "name": "skeleton-to-swc",
            "slurm_kwargs": skeleton_to_swc_slurm_kwargs,
            "slurm_commands": skeleton_2_swc_command_list,
            "job_file": skeleton_2_swc_job_file,
            "start_condition": 'afterok',
        },

        {
            "id": 5,
            "parent_id": 4,
            "name": "cleanup",
            "slurm_kwargs": cleanup_kwargs,
            "slurm_commands": cleanup_command_list,
            "job_file": cleanup_job_file,
            "start_condition": "afterany"
        },

    ]

    pipeline_dag = Slurm_DAG(slurm_dag_node_list)
    last_job_id = pipeline_dag.submit_dag_to_scheduler(parent_job_id=parent_job_id,
                                                       start_condition=start_condition)
    return last_job_id
