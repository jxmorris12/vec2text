from datetime import datetime
import itertools

from slurmpy import Slurm


BASE_PYTHON_CMD = """
python run.py --per_device_train_batch_size {batch_size} \
--max_seq_length {max_seq_length} \
--model_name_or_path {model_name} \
--embedding_model_name {emb_model_name} \
--num_repeat_tokens {num_repeat_tokens} \
--embedder_no_grad {embedder_no_grad} \
--exp_name {exp_name} \
--learning_rate {learning_rate} \
--
--max_eval_samples 400 \
--eval_steps 8000 \
--warmup_steps 4000 \
--bf16=1 \
--use_wandb=1
"""


models = [
    # 't5-small',
    't5-base',
    # 't5-large',
    # 't5-3b',
    # 't5-11b',
]

# emb_models = ['dpr', 'ance_tele']
emb_models = ['dpr']
# emb_models = ["gtr_base", "gtr_large"]


##########################################
# exp_name = 'feb27-t5-size'
# exp_name = 'feb27-token-num-3'
# exp_name = 'feb28-emb'
# exp_name = 'mar1-msl-eng'
# exp_name = 'mar2-gtr'
exp_name = 'mar3-freeze'
##########################################

batch_size = 128
# batch_size = 32

# max_seq_length = [1+1, 4+1, 8+1, 64+1]
max_seq_length = [64]

# embedder_no_grad = [True, False]
embedder_no_grad = [True]

# learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
learning_rates = [5e-4]

num_repeat_tokens = [16]
# num_repeat_tokens = [1, 2, 4, 8, 16, 32, 64, 128]


ACTUALLY_RUN_COMMAND = False

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
for args in itertools.product(
        models, emb_models, learning_rates,
        num_repeat_tokens, max_seq_length, embedder_no_grad
    ):
    m, e, lr, n, msl, eng = args
    total += 1
    cmd = BASE_PYTHON_CMD.format(
        batch_size=batch_size,
        max_seq_length=msl,
        # 
        model_name=m, 
        emb_model_name=e,
        num_repeat_tokens=n,
        learning_rate=lr,
        # 
        exp_name=exp_name,
        embedder_no_grad=eng
    )
    job_desc = ".".join(map(str, args))
    run_cmd(cmd, job_desc=job_desc)


if ACTUALLY_RUN_COMMAND:
    print(f"successfully queued {total} jobs.")
else:
    print(f"successfully queued {total} jobs. (pretend)")

