from datetime import datetime
import itertools

from slurmpy import Slurm


BASE_PYTHON_CMD = """
python run.py --per_device_train_batch_size {batch_size} \
--max_seq_length {max_seq_length} \
--model_name_or_path {model_name} \
--embedding_model_name {emb_model_name} \
--num_repeat_tokens {num_repeat_tokens} \
--exp_name {exp_name} \
--max_eval_samples 200 \
--eval_steps 6000 \
--use_wandb=1
"""


models = [
    # 't5-small',
    't5-base',
    # 't5-large',
    # 't5-3b',
    # 't5-11b',
]

emb_models = [
    'dpr',
]

# exp_name = 'feb27-t5-size'
exp_name = 'feb27-token-num-3'

batch_size = 32
# batch_size = 16
max_seq_length = 128

# num_repeat_tokens = [32]
num_repeat_tokens = [1, 2, 4, 8, 16, 32, 64, 128]


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
for m, e, n in itertools.product(models, emb_models, num_repeat_tokens):
    total += 1
    cmd = BASE_PYTHON_CMD.format(
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        # 
        model_name=m, 
        emb_model_name=e,
        num_repeat_tokens=n,
        # 
        exp_name=exp_name,
    )
    job_desc = ".".join((e, m))
    run_cmd(cmd, job_desc=job_desc)


if ACTUALLY_RUN_COMMAND:
    print(f"successfully queued {total} jobs.")
else:
    print(f"successfully queued {total} jobs. (pretend)")

