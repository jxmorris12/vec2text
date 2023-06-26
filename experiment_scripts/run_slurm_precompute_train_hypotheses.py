import math
import itertools
from datetime import datetime

from slurmpy import Slurm

BASE_PYTHON_CMD = """
python precompute_train_hypotheses.py \
--start_idx {start_idx} \
--num_samples {num_samples}
"""



ACTUALLY_RUN_COMMAND = True
def run_cmd(cmd: str, job_desc: str):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    job_name = f"{dt_string} {job_desc}"
    ##
    print("job_name >>", job_name)
    print("cmd >>", cmd.strip())
    ##

    if ACTUALLY_RUN_COMMAND:
        slurm = Slurm(
            job_name,
            slurm_kwargs={
                "partition": "gpu",
                "gres": "gpu:1",
                "constraint": "gpu-high|gpu-mid",
                "ntasks": 1,
                "cpus-per-task": 4,
                "mem": "48G",
                "time": "96:00:00",  # 168 hours --> 1 week
            },
            slurm_flags=[
                "requeue",
            ],
        )
        slurm.run(
            f"""
        {cmd}
        """
        )
    ##
    print("\n\n")

MSMARCO_LENGTH = 8_753_404
N_SHARDS = 64

shard_length = math.ceil(MSMARCO_LENGTH / N_SHARDS)
start_idxs = []

for i in range(N_SHARDS):
    start_idx = i * shard_length
    cmd = BASE_PYTHON_CMD.format(
        start_idx = i * shard_length,
        num_samples = shard_length,
    )
    cmd = cmd.replace("\n", " ")
    job_desc = ".".join(map(str, ["msmarco_precompute", start_idx, shard_length]))
    run_cmd(cmd, job_desc=job_desc)


if ACTUALLY_RUN_COMMAND:
    print(f"successfully queued {N_SHARDS} jobs.")
else:
    print(f"successfully queued {N_SHARDS} jobs. (pretend)")
