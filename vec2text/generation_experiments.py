import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description="Script for running a model")

# Add arguments
parser.add_argument("--alias", type=str, help="Model alias", required=True)
parser.add_argument(
    "--generation-strategy",
    type=str,
    choices=["reranking", "contrastive", "none"],
    help="Strategy for generation",
    required=True,
)
parser.add_argument(
    "--eval-dataset",
    type=str,
    default="natural_questions",
    help="Dataset for evaluation (default: natural_questions)",
)
parser.add_argument(
    "--overwrite-cache",
    action="store_true",
    default=False,
    help="Whether to overwrite results cache",
)

# Parse the arguments
args = parser.parse_args()
