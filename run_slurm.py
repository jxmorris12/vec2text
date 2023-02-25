from datetime import datetime
import itertools

from slurmpy import Slurm


BASE_PYTHON_CMD = """
python run.py --dataset={dataset} --num_distractor_words={num_distractor_words} --model_name {model_name}
"""

num_distractor_words = [
    0, 4, 16, 64, 256
]

models = ['dpr', 'laprador']

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
                "partition": "rush,gpu",
                "gres": "gpu:a6000:1",
                # "gres": "gpu:1",
                # "constraint": "a40|3090|a6000|a5000|a100-40",
                "ntasks": 1,
                "cpus-per-task": 4,
                "mem": "48G",
                "time": "72:00:00",
            },
            slurm_flags=[
                "requeue",
            ],
        )
        slurm.run(f"""
        {cmd}
        """)
    ##
    print("\n\n")


total = 0
for d, n, m in itertools.product(
        datasets, num_distractor_words, models
    ):
    total += 1
    cmd = BASE_PYTHON_CMD.format(
        dataset=d, num_distractor_words=n, model_name=m
    )
    job_desc = ".".join((d, str(n), m))
    run_cmd(cmd, job_desc=job_desc)


print(f"successfully queued {total} jobs.")
