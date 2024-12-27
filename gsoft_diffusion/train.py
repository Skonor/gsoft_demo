import argparse
from gsoft.trainer import trainers

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument("--trainer_type", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=2000)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--project_name", type=str, default='gsoft')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, default=None, required=True)
    parser.add_argument("--test_data_dir", type=str, default=None, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stable-diffusion-2-base')
    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, required=True)
    parser.add_argument("--prompt_channel", type=int, default=1024)
    parser.add_argument("--finetune_unet", action='store_true', default=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_val_imgs", type=int, default=5)
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--with_prior_preservation", default=False, action="store_true", help="Flag to add prior preservation loss."
    )

    parser.add_argument("--gsoft_nblocks", type=int, default=16)
    parser.add_argument("--gsoft_scale", action='store_true', default=True)
    parser.add_argument("--gsoft_method", type=str, default='cayley')

    return parser.parse_args()


def main(args):
    trainer = trainers[args.trainer_type](args)
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
