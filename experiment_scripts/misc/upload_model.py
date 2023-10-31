"""Uploads a model from local alias (saved on my computer) to a pre-trained
HuggingFace model.

Example usage:
    >> python scripts/upload_model.py openai_msmarco__msl128__200epoch__correct vec2text__openai_ada002__msmarco__msl128__corrector

and you'll have to paste in a HuggingFace.co token that has write access.
 """
import argparse

from huggingface_hub import login as huggingface_login

from vec2text.aliases import load_model_from_alias

# huggingface_login()


def main():
    parser = argparse.ArgumentParser(description="Alias Converter")

    parser.add_argument("alias", type=str, help="Local alias")
    parser.add_argument("new_alias", type=str, help="Model alias for HuggingFace")

    args = parser.parse_args()

    model = load_model_from_alias(args.alias)
    model.push_to_hub(args.new_alias, max_shard_size="200MB")


if __name__ == "__main__":
    main()
