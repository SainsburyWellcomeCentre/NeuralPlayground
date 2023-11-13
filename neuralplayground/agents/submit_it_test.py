import submitit
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import submitit

def add(a, b):
    return a + b

# the AutoExecutor class is your interface for submitting function to a cluster or run them locally.
# The specified folder is used to dump job information, logs and result when finished
# %j is replaced by the job id at runtime
log_folder = "log_test/%j"
a = [1, 2, 3, 4]
b = [10, 20, 30, 40]
executor = submitit.AutoExecutor(folder=log_folder)
# the following line tells the scheduler to only run\
# at most 2 jobs at once. By default, this is several hundreds
executor.update_parameters(slurm_array_parallelism=2)
jobs = executor.map_array(add, a, b)  # just a list of jobs

# waits for completion and returns output
 # 5 + 7 = 12...  your addition was computed in the cluster