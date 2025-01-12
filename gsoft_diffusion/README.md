## GSOFT Stable Diffusion fine-tuning
This directory contains all the code for Stable Diffusion fine-tuning experiments.

## Running locally with PyTorch
### Installing the dependencies

Before running the scripts, make sure to install training dependencies:

```bash
pip install -r requirements.txt
```

And initialize an [🤗 Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

### Dog toy example

Now let's get the training dataset. For this example we will use some dog images: https://huggingface.co/datasets/diffusers/dog-example.

Let's first download it locally:

```py
from huggingface_hub import snapshot_download

local_dir = "dog"
snapshot_download("diffusers/dogexample", local_dir=local_dir, repo_type="dataset", ignore_patterns=".gitattributes")
```

Now the training can be launched using:

```bash
export MODEL_NAME="stable-diffusion-2-base"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --gsoft_nblocks=64 \
  --gsoft_method="cayley" \
  --seed=0 \
  --gsoft_scale \
  --double_gsoft
```

### Inference

Once you have trained a model using above command, the inference can be done using the following script. The 'prompts' argument should be passed as a one string of prompt devided by #.

```bash
export MODEL_NAME="stable-diffusion-2-base"
export INPUT_DIR="path-to-saved-model"
export OUTPUT_DIR="path-to-save-inference"

accelerate launch inference.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --input_dir=$INPUT_DIR \
  --checkpoint_idx=2000 \
  --prompts="a photo of sks dog#sks dog in a purple wizzard outfit#sks dog with mountains in the background" \
  --num_images_per_prompt=3 \
  --batch_size=3 \
  --gsoft_nblocks=64 \
  --gsoft_method="cayley" \
  --seed=0 \
  --gsoft_scale \
  --double_gsoft
```

Then the result will be saved in the following structure:

```md
path-to-save-inference
├── a photo of sks dog
│   ├── 0.png
│   ├── 1.png
│   └── 2.png
├── sks dog in a purple wizzard outfit
│   ├── 0.png
│   ├── 1.png
│   └── 2.png
├── sks dog with mountains in the background
│   ├── 0.png
│   ├── 1.png
│   └── 2.png
```