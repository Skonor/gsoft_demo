import argparse
import yaml

from gsoft.inferencer import inferencers

import warnings
warnings.filterwarnings('ignore')


#  please, change the evaluation sets if required
base_set = [
    "a photo of a {0}"
]

evaluation_set = [
    'a {0} with the Eiffel Tower in the background',
    'a {0} with a mountain in the background',
    'a {0} in the space'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to hparams.yml"
    )
    parser.add_argument(
        "--checkpoint_idx",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_images_per_medium_prompt",
        type=int,
        default=10,
        help="Number of generated images for each medium prompt",
    )
    parser.add_argument(
        "--num_images_per_base_prompt",
        type=int,
        default=30,
        help="Number of generated images for each base prompt",
    )
    parser.add_argument(
        "--batch_size_medium",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size_base",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0
    )
    parser.add_argument(
        "--replace_inference_output",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--version",
        type=int,
        default=0
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    return parser.parse_args()


def main(args):
    with open(args.config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    inferencer = inferencers[args.inference_type](config, args, evaluation_set, base_set)

    inferencer.setup()
    inferencer.generate()


if __name__ == '__main__':
    args = parse_args()
    main(args)
