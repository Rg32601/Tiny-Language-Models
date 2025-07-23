# configs/

This folder contains configuration and documentation files for each pretrained Tiny-Language-Model release.

## What’s Here

- Each `.yml` file documents a specific model release tag.
- These files include architecture details, training settings, and usage notes.
- You can use the tags to automatically download and load models via utility functions.

## How to Use

### 1. Clone the Repo
  ```python
    !git clone https://github.com/Rg32601/Tiny-Language-Models
  ```
### 2. Load a Pre-trained Model
- Use the tag from the desired YAML file to load the matching model:
  ```python
  from configs.utils import download_and_load_pre_trained_model
  
  model_tag = "<pretrained_model_tag>"  
  model = download_and_load_pre_trained_model(model_tag)
  ```
### 3. Load a Fine-tuned Model
  ```python
  from configs.utils import download_and_load_finetuned_model

  model_tag = "<finetune_model_tag>"  # e.g. finetune-tinybert-layers6-conv2-k3-d64-heads12-maxlen128-vocab30522-prewiki90000-ftdatasetagnews-labels4-ftep10
  model = download_and_load_finetuned_model(model_tag)
  ```
> Replace `<pretrained_model_tag>` or `<finetune_model_tag>` with the actual tag you want to use.  
> Tags are specified in the corresponding YAML files in this folder.

**Notes:**
- The loader automatically downloads the model files from the GitHub release if they’re not already present locally.
- Model architecture details are parsed from the tag or YAML.
- The returned `model` object is ready for inference or further training.

