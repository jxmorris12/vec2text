import aliases
import os

def precompute(start_idx: int, num_samples: int):
    # load the previously-trained msmarco model 
    exp, trainer = aliases.load_experiment_and_trainer_from_alias(
        "openai_msmarco__msl128__100epoch__correct",
        max_seq_length=128,
        use_less_data=-1,
    )

    end_idx = min(len(trainer.train_dataset), start_idx + num_samples)
    print("Original length:", len(trainer.train_dataset))
    trainer.train_dataset = trainer.train_dataset.select(range(start_idx, end_idx))
    print("Sampled length:", len(trainer.train_dataset))
    hypothesis_path = trainer.precompute_hypotheses()
    out_path = f"/home/jxm3/research/retrieval/inversion/msmarco_msl128_hypotheses/msmarco_{num_samples}_{start_idx}.arrow"
    os.symlink(hypothesis_path, out_path)
    print(f"precomputed {num_samples} samples from msmarco from idx {start_idx} and saved to {out_path}")


def gather():
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="precompute MSMARCO hypotheses")
    parser.add_argument("--start_idx", type=int, required=True, help="Starting index")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples")
    parser.add_argument("--work", type=str, 
        required=False, default="precompute", 
        choices=["precompute", "gather"],
        help="type of work to do"
    )

    args = parser.parse_args()

    if args.work == "precompute":
        precompute(args.start_idx, args.num_samples)
    else:
        gather()