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
python evaluate_models.py {alias} \
--num_samples 500 \
--return_best_hypothesis {return_best_hypothesis} \
--num_gen_recursive_steps {num_gen_recursive_steps} \
--sequence_beam_width {sequence_beam_width} \
--beam_width {beam_width}
"""

ACTUALLY_RUN_COMMAND = True
def run_command(cmd: str, job_desc: str):
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
                "ntasks": 1,
                "cpus-per-task": 4,
                "mem": "48G",
                "time": "48:00:00",
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
    print("\n")


def main():
    for alias in inv_aliases:
        command = base_command.format(
            alias=alias,
            return_best_hypothesis=False,
            num_gen_recursive_steps=1,
            sequence_beam_width=1,
            beam_width=1,
        )
        run_command(command, "evaluate_inversion")
    
    for alias in corr_aliases:
        # single step
        command = base_command.format(
            alias=alias,
            return_best_hypothesis=False,
            num_gen_recursive_steps=1,
            sequence_beam_width=1,
            beam_width=1,
        )
        run_command(command, "evaluate_corrector")
        for num_steps in [10, 50]:
            for return_best_hypothesis in [0, 1]:
                # 10 steps
                command = base_command.format(
                    alias=alias,
                    return_best_hypothesis=return_best_hypothesis,
                    num_gen_recursive_steps=num_steps,
                    sequence_beam_width=1,
                    beam_width=1,
                )
                run_command(command, "evaluate_corrector")
                # 10 steps (sbw 8)
                command = base_command.format(
                    alias=alias,
                    return_best_hypothesis=return_best_hypothesis,
                    num_gen_recursive_steps=num_steps,
                    sequence_beam_width=8,
                    beam_width=1,
                )
                run_command(command, "evaluate_corrector")
                # 10 steps (sbw 8)
                command = base_command.format(
                    alias=alias,
                    return_best_hypothesis=return_best_hypothesis,
                    num_gen_recursive_steps=num_steps,
                    sequence_beam_width=1,
                    beam_width=8,
                )
                run_command(command, "evaluate_corrector")


if __name__ == '__main__':
    main()