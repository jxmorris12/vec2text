import argparse
import json

import aliases

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Argument Parser')

    parser.add_argument('alias', type=str, help='Trained model alias from alias.py')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of evaluation samples')
    parser.add_argument('--return_best_hypothesis', type=bool, default=False, help='Whether to return best hypothesis during generation')
    parser.add_argument('--num_gen_recursive_steps', type=int, default=1, help='Number of steps for recursive generation')
    parser.add_argument('--sequence_beam_width', type=int, default=1, help='Sequence-level beam width')
    parser.add_argument('--beam_width', type=int, default=1, help='Regular beam width')

    return parser


def main(args: argparse.ArgParser):
    experiment, trainer = (
        aliases.load_experiment_and_trainer_from_alias(args.alias)
    )
    trainer.model.eval()
    trainer.args.per_device_eval_batch_size = 1
    trainer.return_best_hypothesis = True
    trainer.num_gen_recursive_steps = 15
    trainer.sequence_beam_width = 16
    trainer.gen_kwargs = {
        "early_stopping": False,
        "num_beams": args.beam_width,
        "num_return_sequences": args.beam_width,
        "do_sample": False,
        "no_repeat_ngram_size": 0,
    }
    metrics = trainer.evaluate(
        eval_dataset=inv_trainer.eval_dataset[experiment.data_args.dataset].select(range(args.num_samples))
    )


if __name__ == '__main__':
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    main(args=args)