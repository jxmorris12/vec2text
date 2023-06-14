from datetime import datetime
from slurmpy import Slurm


inv_aliases = [
    "dpr_nq__msl32_beta",
    "openai_msmarco__msl128__100epoch",
]

corr_aliases = [
    "gtr_nq__msl32_beta__correct",
    "openai_msmarco__msl128__100epoch__correct",
    "openai_msmarco__msl128__100epoch__correct_cheat",
]

base_command = """
python evaluate.py {alias} \
--num_samples 500 \
--return_best_hypothesis {return_best_hypothesis} \
--num_gen_recursive_steps {num_gen_recursive_steps} \
--sequence_beam_width {sequence_beam_width} \
--beam_width {beam_width}
"""

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
                "partition": "rush",
                "gres": "gpu:a6000:1",
                # "gres": "gpu:1",
                # "constraint": "a40|3090|a6000|a5000|a100-40",
                "ntasks": 1,
                "cpus-per-task": 4,
                "mem": "48G",
                # "nodelist": "rush-compute-03",
                # "time": "24:00:00",
                # "time": "72:00:00",
                "time": "168:00:00",  # 168 hours --> 2 weeks
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


def main():
    for alias in inv_aliases:
        command = base_command.format(
            alias=alias,
            return_best_hypothesis=False,
            num_gen_recursive_steps=1,
            sequence_beam_width=1,
            beam_width=1,
        )
        run_command(command)
    
    for alias in corr_aliases:
        # single step
        command = base_command.format(
            alias=alias,
            return_best_hypothesis=False,
            num_gen_recursive_steps=1,
            sequence_beam_width=1,
            beam_width=1,
        )
        run_command(command)
        # 10 steps
        command = base_command.format(
            alias=alias,
            return_best_hypothesis=False,
            num_gen_recursive_steps=10,
            sequence_beam_width=1,
            beam_width=1,
        )
        run_command(command)
        # 10 steps (sbw 8)
        command = base_command.format(
            alias=alias,
            return_best_hypothesis=False,
            num_gen_recursive_steps=1,
            sequence_beam_width=8,
            beam_width=1,
        )
        run_command(command)
        # 10 steps (sbw 8)
        command = base_command.format(
            alias=alias,
            return_best_hypothesis=False,
            num_gen_recursive_steps=1,
            sequence_beam_width=1,
            beam_width=8,
        )
        run_command(command)
        # 50 steps
        command = base_command.format(
            alias=alias,
            return_best_hypothesis=False,
            num_gen_recursive_steps=10,
            sequence_beam_width=1,
            beam_width=1,
        )
        run_command(command)
        # 50 steps (sbw 8)
        command = base_command.format(
            alias=alias,
            return_best_hypothesis=False,
            num_gen_recursive_steps=1,
            sequence_beam_width=8,
            beam_width=1,
        )
        run_command(command)
        # 50 steps (sbw 8)
        command = base_command.format(
            alias=alias,
            return_best_hypothesis=False,
            num_gen_recursive_steps=1,
            sequence_beam_width=1,
            beam_width=8,
        )
        run_command(command)


if __name__ == '__main__':
    main()