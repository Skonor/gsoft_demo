import os
from typing import Callable, Any, Optional
import yaml
import secrets
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import diffusers
from diffusers import (
    AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import transformers
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from .model.gsoft import GSOFTCrossAttnProcessor, DoubleGSOFTCrossAttnProcessor
from .data.dataset import ImageDataset, DreamBoothDataset, collate_fn
from .utils.registry import ClassRegistry


logger = get_logger(__name__)
trainers = ClassRegistry()


@trainers.add_to_registry('base')
class BaseTrainer:
    def __init__(self, config):
        self.config = config

    def setup_exp_name(self, exp_idx):
        exp_name = '{0:0>5d}-{1}-{2}'.format(
            exp_idx + 1, secrets.token_hex(2), os.path.basename(os.path.normpath(self.config.test_data_dir))
        )
        exp_name += '_qkv'
        return exp_name

    def setup_exp(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        exp_idx = 0
        for folder in os.listdir(self.config.output_dir):
            try:
                curr_exp_idx = max(exp_idx, int(folder.split('-')[0].lstrip('0')))
                exp_idx = max(exp_idx, curr_exp_idx)
            except:
                pass

        self.exp_name = self.setup_exp_name(exp_idx)
        self.config.exp_name = self.exp_name
        self.output_dir = os.path.abspath(os.path.join(self.config.output_dir, self.exp_name))
        self.config.output_dir = self.output_dir
        self.logging_dir = os.path.join(self.output_dir, 'logs')
        self.config.logging_dir = self.logging_dir

    def setup_accelerator(self):
        accelerator_project_config = ProjectConfiguration(project_dir=self.output_dir)
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            project_config=accelerator_project_config,
        )
        self.accelerator.init_trackers(
            project_name=self.config.project_name,
            config=self.config,
        )
        os.makedirs(self.logging_dir, exist_ok=True)
        with open(os.path.join(self.logging_dir, "hparams.yml"), "w") as outfile:
            yaml.dump(vars(self.config), outfile)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def setup_base_model(self):
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="unet"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="vae", revision=self.config.revision
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer", revision=self.config.revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.config.revision
        )

    def setup_model(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.params_to_optimize = []
        if self.config.finetune_unet:
            self.unet.train()
            for (name, param) in self.unet.named_parameters():
                if 'to_q' in name or 'to_k' in name or 'to_v' in name or 'to_out.0' in name:
                    param.requires_grad = True
                    self.params_to_optimize.append(param)

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.params_to_optimize,
            lr=2e-5,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )

    def setup_lr_scheduler(self):
        pass

    def setup_dataset(self):
        if self.config.with_prior_preservation:
            self.train_dataset = DreamBoothDataset(
                instance_data_root=self.config.train_data_dir,
                instance_prompt="a photo of a {0} {1}".format(self.config.placeholder_token, self.config.class_name),
                class_data_root=self.config.class_data_dir if self.config.with_prior_preservation else None,
                class_prompt="a photo of a {0}".format(self.config.class_name),
                tokenizer=self.tokenizer,
            )
            collator: Optional[Callable[[Any], dict[str, torch.Tensor]]] = lambda examples: collate_fn(
                examples, self.config.with_prior_preservation
            )
        else:
            self.train_dataset = ImageDataset(
                train_data_dir=self.config.train_data_dir
            )
            collator = None

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.dataloader_num_workers,
        )

    def move_to_device(self):
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.optimizer, self.train_dataloader = self.accelerator.prepare(self.optimizer, self.train_dataloader)
        if self.config.finetune_unet:
            self.unet = self.accelerator.prepare(self.unet)

        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

    def setup(self):
        self.setup_exp()
        self.setup_accelerator()
        self.setup_base_model()
        self.setup_model()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.setup_dataset()
        self.move_to_device()
        self.setup_pipeline()

    def train_step(self, batch):
        if self.config.with_prior_preservation:
            latents = self.vae.encode(batch['pixel_values']).latent_dist.sample()
        else:
            latents = self.vae.encode(batch['image'] * 2.0 - 1.0).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=latents.device)

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        if not self.config.with_prior_preservation:
            input_ids = self.tokenizer(
                "a photo of {0} {1}".format(self.config.placeholder_token, self.config.class_name),
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )["input_ids"]
            encoder_hidden_states = self.text_encoder(input_ids.to(device=latents.device))[0]
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(latents.shape[0], 0)
        else:
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
        outputs = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if self.config.with_prior_preservation:
            outputs, prior_outputs = torch.chunk(outputs, 2, dim=0)
            target, prior_target = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = torch.nn.functional.mse_loss(outputs.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = torch.nn.functional.mse_loss(prior_outputs.float(), prior_target.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + self.config.prior_loss_weight * prior_loss
        else:
            loss = torch.nn.functional.mse_loss(outputs.float(), target.float(), reduction="mean")

        return loss

    def setup_pipeline(self):
        _scheduler = DDIMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            scheduler=_scheduler,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            unet=self.accelerator.unwrap_model(self.unet),
            vae=self.vae,
            revision=self.config.revision,
            torch_dtype=torch.float16,
            requires_safety_checker=False,
        )
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def prepare_prompts(self, prompts):
        return prompts

    def save_model(self, epoch):
        save_path = os.path.join(self.output_dir, f"checkpoint-{epoch}")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        if self.config.finetune_unet:
            torch.save(self.unet.state_dict(), os.path.join(save_path, 'unet.bin'))

    def train(self):
        for epoch in tqdm(range(self.config.num_train_epochs)):
            batch = next(iter(self.train_dataloader))
            loss = self.train_step(batch)
            for tracker in self.accelerator.trackers:
                tracker.log({"loss": loss})
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.accelerator.is_main_process:
                if epoch in [500, 1000, 3000]:
                    self.save_model(epoch)
        self.save_model(self.config.num_train_epochs)

        self.accelerator.end_training()


@trainers.add_to_registry('gsoft')
class GSOFTTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def setup_exp_name(self, exp_idx):
        exp_name = '{0:0>5d}-{1}-{2}'.format(
            exp_idx + 1, secrets.token_hex(2), os.path.basename(os.path.normpath(self.config.test_data_dir))
        )
        exp_name += f'_gsoft{self.config.gsoft_nblocks}'
        if self.config.gsoft_scale:
            exp_name += '_scale'
        return exp_name

    def setup_model(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.params_to_optimize = []
        gsoft_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            gsoft_attn_procs[name] = GSOFTCrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config.gsoft_nblocks,
                method=self.config.gsoft_method, scale=self.config.gsoft_scale
            )

        self.unet.set_attn_processor(gsoft_attn_procs)
        self.gsoft_layers = AttnProcsLayers(self.unet.attn_processors)
        self.accelerator.register_for_checkpointing(self.gsoft_layers)
        for (name, param) in self.gsoft_layers.named_parameters():
            param.requires_grad = True
            self.params_to_optimize.append(param)
        self.gsoft_layers.train()

    def setup_lr_scheduler(self):
        self.lr_scheduler = get_scheduler(
            "constant",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.config.num_train_epochs,
            num_cycles=1,
            power=1.0,
        )

    def move_to_device(self):
        self.gsoft_layers, self.optimizer, self.train_dataloader = self.accelerator.prepare(self.gsoft_layers, self.optimizer, self.train_dataloader)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

    def save_model(self, epoch):
        save_path = os.path.join(self.output_dir, f"checkpoint-{epoch}")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        self.unet.save_attn_procs(os.path.join(save_path))

    def train(self):
        for epoch in tqdm(range(self.config.num_train_epochs)):
            batch = next(iter(self.train_dataloader))
            loss = self.train_step(batch)
            for tracker in self.accelerator.trackers:
                tracker.log({"loss": loss})
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.gsoft_layers.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.accelerator.is_main_process:
                if epoch in [500, 1000, 3000]:
                    self.save_model(epoch)
        self.save_model(self.config.num_train_epochs)

        self.accelerator.end_training()


@trainers.add_to_registry('double_gsoft')
class DoubleGSOFTTrainer(GSOFTTrainer):
    def __init__(self, config):
        super().__init__(config)

    def setup_exp_name(self, exp_idx):
        exp_name = '{0:0>5d}-{1}-{2}'.format(
            exp_idx + 1, secrets.token_hex(2), os.path.basename(os.path.normpath(self.config.test_data_dir))
        )
        exp_name += f'_doublegsoft{self.config.gsoft_nblocks}'
        if self.config.gsoft_scale:
            exp_name += '_scale'
        return exp_name

    def setup_model(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.params_to_optimize = []
        gsoft_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            gsoft_attn_procs[name] = DoubleGSOFTCrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config.gsoft_nblocks,
                method=self.config.gsoft_method, scale=self.config.gsoft_scale
            )

        self.unet.set_attn_processor(gsoft_attn_procs)
        self.gsoft_layers = AttnProcsLayers(self.unet.attn_processors)
        self.accelerator.register_for_checkpointing(self.gsoft_layers)
        for (name, param) in self.gsoft_layers.named_parameters():
            param.requires_grad = True
            self.params_to_optimize.append(param)
        self.gsoft_layers.train()
