import os
from tqdm import tqdm
import random
import numpy as np

import torch
from diffusers import (
    StableDiffusionPipeline, AutoencoderKL, DDIMScheduler, UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file

from .model.gsoft import GSOFTCrossAttnProcessor, DoubleGSOFTCrossAttnProcessor
from .utils.registry import ClassRegistry


inferencers = ClassRegistry()


def get_seed(prompt, i, seed):
    h = 0
    for el in prompt:
        h += ord(el)
    h += i
    return h + seed


@inferencers.add_to_registry('base')
class BaseInferencer:
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):
        self.config = config
        self.args = args
        self.checkpoint_idx = args.checkpoint_idx
        self.num_images_per_context_prompt = args.num_images_per_medium_prompt
        self.num_images_per_base_prompt = args.num_images_per_base_prompt
        self.batch_size_context = args.batch_size_medium
        self.batch_size_base = args.batch_size_base

        if self.checkpoint_idx is None:
            self.checkpoint_path = config['output_dir']
        else:
            self.checkpoint_path = os.path.join(config['output_dir'], f'checkpoint-{self.checkpoint_idx}')

        self.context_prompts = context_prompts
        self.base_prompts = base_prompts

        self.replace_inference_output = self.args.replace_inference_output
        self.version = self.args.version

        self.device = device
        self.dtype = dtype

    def setup_pipe_kwargs(self):
        self.pipe_kwargs = {
            'guidance_scale': self.args.guidance_scale,
            'num_inference_steps': self.args.num_inference_steps,
        }

    def setup_base_model(self):
        # Here we create base models
        self.scheduler = DDIMScheduler.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="scheduler"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config['pretrained_model_name_or_path'], subfolder="unet"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="vae", revision=self.config['revision']
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="tokenizer", revision=self.config['revision']
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            subfolder="text_encoder", revision=self.config['revision']
        )

    def setup_model(self):
        self.unet.load_state_dict(torch.load(
            os.path.join(self.checkpoint_path, 'unet.bin')
        ))

    def setup_pipeline(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config['pretrained_model_name_or_path'],
            scheduler=self.scheduler,
            tokenizer=self.tokenizer,
            unet=self.unet,
            vae=self.vae,
            text_encoder=self.text_encoder,
            revision=None,
            requires_safety_checker=False,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

    def setup(self):
        self.setup_base_model()
        self.setup_model()
        self.setup_pipeline()
        self.setup_pipe_kwargs()
        self.create_folder_name()
        self.setup_paths()

    def prepare_prompts(self, context_prompts, base_prompts):
        return context_prompts, base_prompts

    def create_folder_name(self):
        self.inference_folder_name = f"ns{self.args.num_inference_steps}_gs{self.args.guidance_scale}"

    def setup_paths(self):
        if self.version is None:
            version = 0
            samples_path = os.path.join(
                self.checkpoint_path, 'samples', self.inference_folder_name, f'version_{version}'
            )
            if os.path.exists(samples_path):
                while not os.path.exists(samples_path):
                    samples_path = os.path.join(
                        self.checkpoint_path, 'samples', self.inference_folder_name, f'version_{version}'
                    )
                    version += 1
        else:
            samples_path = os.path.join(
                self.checkpoint_path, 'samples', self.inference_folder_name, f'version_{self.version}'
            )
        self.samples_path = samples_path

    def check_generation(self, path, num_images_per_prompt):
        if self.replace_inference_output:
            return True
        else:
            if os.path.exists(path) and len(os.listdir(path)) == num_images_per_prompt:
                return False
            else:
                return True

    def generate_with_prompt(self, prompt, num_images_per_prompt, batch_size):
        n_batches = (num_images_per_prompt - 1) // batch_size + 1
        images = []
        for i in range(n_batches):
            seed = get_seed(prompt, i, self.args.seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            generator = torch.Generator(device='cuda')
            generator = generator.manual_seed(seed)
            images_batch = self.pipe(
                prompt=prompt.format(f"{self.config['placeholder_token']} {self.config['class_name']}"),
                generator=generator, num_images_per_prompt=batch_size, **self.pipe_kwargs
            ).images
            images += images_batch
        return images

    def save_images(self, images, path):
        os.makedirs(path, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(os.path.join(path, f'{idx}.png'))

    def generate_with_prompt_list(self, prompts, num_images_per_prompt, batch_size):
        for prompt in tqdm(prompts):
            formatted_prompt = prompt.format(self.config['placeholder_token'])
            path = os.path.join(self.samples_path, formatted_prompt)
            if self.check_generation(path, num_images_per_prompt):
                images = self.generate_with_prompt(
                    prompt, num_images_per_prompt, batch_size)
                self.save_images(images, path)

    def generate(self):
        context_prompts, base_prompts = self.prepare_prompts(self.context_prompts, self.base_prompts)
        self.generate_with_prompt_list(
            context_prompts, self.num_images_per_context_prompt, self.batch_size_context)
        self.generate_with_prompt_list(
            base_prompts, self.num_images_per_base_prompt, self.batch_size_base)


@inferencers.add_to_registry('gsoft')
class GSOFTInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)

    def setup_model(self,):
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
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['gsoft_nblocks'],
                method=self.config['gsoft_method'], scale=self.config['gsoft_scale']
            )

        self.unet.set_attn_processor(gsoft_attn_procs)
        self.gsoft_layers = AttnProcsLayers(self.unet.attn_processors)
        self.gsoft_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights.safetensors')))


@inferencers.add_to_registry('double_gsoft')
class DoubleGSOFTInferencer(BaseInferencer):
    def __init__(self, config, args, context_prompts, base_prompts,
                 dtype=torch.float32, device='cuda'):

        super().__init__(config, args, context_prompts, base_prompts, dtype, device)

    def setup_model(self,):
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
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, nblocks=self.config['gsoft_nblocks'],
                method=self.config['gsoft_method'], scale=self.config['gsoft_scale']
            )

        self.unet.set_attn_processor(gsoft_attn_procs)
        self.gsoft_layers = AttnProcsLayers(self.unet.attn_processors)
        self.gsoft_layers.load_state_dict(load_file(os.path.join(self.checkpoint_path, 'pytorch_lora_weights.safetensors')))
