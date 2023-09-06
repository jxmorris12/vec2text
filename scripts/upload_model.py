import argparse

from vec2text.run_args import load_model_from_alias

def main():
    parser = argparse.ArgumentParser(description="Alias Converter")

    parser.add_argument("alias", type=str, help="Model alias")
    parser.add_argument("new_alias", type=str, help="Model alias for HuggingFace")

    args = parser.parse_args()

    model = load_model_from_alias(args.alias)
    model.push_to_hub(args.new_alias)
    


if __name__ == "__main__":
    main()