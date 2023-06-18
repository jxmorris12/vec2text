import itertools
from datetime import datetime

from slurmpy import Slurm

all_beir_datasets = [
    ####### public datasets #######
    "arguana",
    "climate-fever",
    "cqadupstack",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "trec-covid",
    "webis-touche2020",
    ####### private datasets #######
    "signal1m",
    "trec-news",
    "robust04",
    "bioasq",
]

model_name = "sentence-transformers/gtr-t5-base"
ACTUALLY_RUN_COMMAND = True

BASE_PYTHON_CMD = """
python scripts/evaluate_retrieval_under_noise.py \
--dataset {dataset} \
--model {model_name} \
--noise_level {noise_level} \
--max_seq_length {max_seq_length}
"""

# noise_levels = [0, 1e-3, 1e-2, 1e-1, 1]  # [0, 1e-3, 1e-2, 1e-1, 1]
# max_seq_lengths = [32]  # [None]
noise_levels = [0]
max_seq_lengths = [128]


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
                # "nodelist": "rush-compute-03",
                # "time": "24:00:00",
                # "time": "72:00:00",
                "time": "168:00:00",  # 168 hours --> 1 week
                # "time": "504:00:00",  # 504 hours --> 3 weeks
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


total = 0

for dataset, noise_level, max_seq_length in itertools.product(
    all_beir_datasets, noise_levels, max_seq_lengths
):
    total += 1
    cmd = BASE_PYTHON_CMD.format(
        dataset=dataset,
        model_name=model_name,
        noise_level=noise_level,
        max_seq_length=max_seq_length,
    )
    run_cmd(cmd=cmd, job_desc="beir_noisy")

if ACTUALLY_RUN_COMMAND:
    print(f"successfully queued {total} jobs.")
else:
    print(f"successfully queued {total} jobs. (pretend)")
