from collections import deque
import subprocess
import os


class InvalidWorkflow(ValueError):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'The workflow graph is invalid'


def validate_slurm_dag(slurm_dag):
    root_ct = len([item for item in slurm_dag.nodes if item['parent_id'] == -1])
    if root_ct != 1:
        raise InvalidWorkflow()

    node_ids = [n['id'] for n in slurm_dag.nodes]
    orphaned_nodes = [n for n in slurm_dag.nodes if (n['parent_id'] not in node_ids) and (n['parent_id'] != -1)]
    if orphaned_nodes:
        raise InvalidWorkflow()


def submit_job_return_id(job_file, parent_job_id, start_condition):
    """
    Will submit a job file with the dependency type (start_condition) on parent_job_id. If a parent_job_id is
    specified, a start condition must also be specified.

    :param start_condition: (str): slurm start condition (e.g. afterok, afterany)
    :param job_file: (str): path to job file to submit
    :param parent_job_id: (int): slurm job id for parent dependency job
    :return:
    """

    accepted_start_condition = ['afterany', 'afterok', None]
    if start_condition not in accepted_start_condition:
        raise ValueError(f"{start_condition} not in accepted start condition {accepted_start_condition}")

    if None in [parent_job_id, start_condition]:
        if not all([i is None for i in [parent_job_id, start_condition]]):
            raise ValueError(
                f"If parent_job_id ({parent_job_id}) or start_conditon ({start_condition}) is defined, both must be "
                f"defined (not None)")

    if parent_job_id:
        command = "sbatch --dependency={}:{} {}".format(start_condition, parent_job_id, job_file)
    else:
        command = "sbatch {}".format(job_file)
    command_list = command.split(" ")
    result = subprocess.run(command_list, stdout=subprocess.PIPE)
    std_out = result.stdout.decode('utf-8')

    job_id = std_out.split("Submitted batch job ")[-1].replace("\n", "")
    print(command)

    return job_id


def create_job_file(dag_node):
    """
    Will create a slurm job file from parameters contained in the dag_node

    :param dag_node: (dict) : keys: job_file, slurm_kwargs, slurm_commands, with the respective value datatypes
    string/path to job file, dict/representing resource requests for slurm, list of str/commands to execute in job
    :return: None
    """
    job_file = dag_node["job_file"]
    slurm_kwargs = dag_node["slurm_kwargs"]
    command_list = dag_node["slurm_commands"]

    job_string_list = [f"#SBATCH {k}={v}" for k, v in slurm_kwargs.items()]
    job_string_list = job_string_list + command_list
    job_string_list = ["#!/bin/bash"] + job_string_list

    if os.path.exists(job_file):
        os.remove(job_file)

    with open(job_file, 'w') as job_f:
        for val in job_string_list:
            job_f.write(val)
            job_f.write('\n')


class Slurm_DAG:
    """
    Slurm workflow DAG to process an individual specimen through various steps of the pipeline
    """
    def __init__(self, list_of_nodes):
        self.nodes = list_of_nodes
        validate_slurm_dag(self)

    def get_root(self):
        return [n for n in self.nodes if n['parent_id'] == -1][0]

    def get_children(self, node):
        return [n for n in self.nodes if n['parent_id'] == node['id']]

    def dfs_traversal(self):
        """Travers the dag in depth first search"""

        queue = deque([self.get_root()])
        dfs_nodes = []
        while len(queue) > 0:
            curr_node = queue.popleft()
            dfs_nodes.append(curr_node)
            children = self.get_children(curr_node)
            for ch in children:
                queue.appendleft(ch)

        return dfs_nodes

    def submit_dag_to_scheduler(self, parent_job_id, start_condition):
        """
        Submit dag to slurm scheduler

        :param parent_job_id: (int): slurm job id dependency for starting this dag
        :param start_condition: (str) start condition for above job dependency (e.g. afterok, afterany)
        :return parent_job_id: (int) the slurm job id associated with the last step in this DAG
        """

        root = self.get_root()

        job_node_list = self.dfs_traversal()
        for node in job_node_list:
            # TODO add validate dag node step
            if node['id'] != root['id']:
                start_condition = node['start_condition']
            create_job_file(node)
            job_file_path = node['job_file']
            parent_job_id = submit_job_return_id(job_file_path, parent_job_id, start_condition)

        return parent_job_id
