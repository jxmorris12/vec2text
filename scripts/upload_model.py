import argparse

from huggingface_hub import login as huggingface_login

from vec2text.aliases import load_model_from_alias

huggingface_login()


def main():
    parser = argparse.ArgumentParser(description="Alias Converter")

    parser.add_argument("alias", type=str, help="Model alias")
    parser.add_argument("new_alias", type=str, help="Model alias for HuggingFace")

    args = parser.parse_args()

    model = load_model_from_alias(args.alias)
    model.push_to_hub(args.new_alias)


if __name__ == "__main__":
    main()
