import sqlite3
import datetime
import os


def status_update(database_path, runs_unique_id, process_name, state):
    """
    Update status of a row in the runs table

    :param runs_unique_id: specimen_runs database unique row id
    :param database_path: path to sqlite .db file
    :param process_name: str, must be one of ['preprocessing', 'segmentation', 'postprocessing', 'stack2swc']
    :param state: str, must be one of ['start', 'finish']
    :return: None

    """
    # get the slurm job id that this process is being run on
    if 'SLURM_JOB_ID' in os.environ.keys():
        slurm_id = os.environ['SLURM_JOB_ID']
    else:
        # if it is not run on slurm just give it a dummy value of 0
        slurm_id = 0
    submission_time = str(datetime.datetime.now())
    status = 1

    time_column = "{}_{}_time".format(process_name, state)
    status_column = "{}_{}".format(process_name, state)
    slurm_id_column = "{}_slurm_job_id".format(process_name)

    if status_column != "pipeline_finish":
        insert_cmd = f"""
        UPDATE specimen_runs
        SET {time_column} = '{submission_time}', {status_column} = {status}, {slurm_id_column} = {slurm_id} 
        WHERE run_id = {runs_unique_id} 
        """
    else:
        # Don't need to keep track of the job-id used to verify the specimen finished running the pipeline
        insert_cmd = f"""
        UPDATE specimen_runs
        SET {status_column} = {status}
        WHERE run_id = {runs_unique_id} 
        """
    print("Executing SQLite Command:")
    print(insert_cmd)
    con = sqlite3.connect(database_path)
    cur = con.cursor()
    cur.execute(insert_cmd)
    con.commit()
    cur.close()
    con.close()


def create_runs_table(database_path):
    """
    Create the specimen_runs table if it does not exist in the .db file provided
    :param database_path: path to sqlite .db file
    :return: None
    """

    create_runs_command = """
    CREATE TABLE IF NOT EXISTS specimen_runs(
    run_id INTEGER PRIMARY KEY,
    specimen_id INTEGER, 
    patchseq_autotrace_version TEXT,
    submit_datetime TEXT,
    output_dirname TEXT DEFAULT NONE,
    preprocessing_start INTEGER DEFAULT 0,
    preprocessing_start_time TEXT DEFAULT NONE,
    preprocessing_finish INTEGER DEFAULT 0,
    preprocessing_finish_time TEXT DEFAULT NONE,
    preprocessing_slurm_job_id INTEGER DEFAULT 0,
    segmentation_start INTEGER DEFAULT 0,
    segmentation_start_time TEXT DEFAULT NONE,
    segmentation_finish INTEGER DEFAULT 0,
    segmentation_finish_time TEXT DEFAULT NONE,
    segmentation_slurm_job_id INTEGER DEFAULT 0,
    postprocessing_start INTEGER DEFAULT 0,
    postprocessing_start_time TEXT DEFAULT NONE,
    postprocessing_finish INTEGER DEFAULT 0,
    postprocessing_finish_time TEXT DEFAULT NONE,
    postprocessing_slurm_job_id INTEGER DEFAULT 0,
    stack2swc_start INTEGER DEFAULT 0,
    stack2swc_start_time TEXT DEFAULT NONE,
    stack2swc_finish INTEGER DEFAULT 0,
    stack2swc_finish_time TEXT DEFAULT NONE,
    stack2swc_slurm_job_id INTEGER DEFAULT 0,
    pipeline_finish INTEGER DEFAULT 0
    );
    """

    con = sqlite3.connect(database_path)
    cur = con.cursor()
    cur.execute(create_runs_command)
    con.commit()
    cur.close()
    con.close()
