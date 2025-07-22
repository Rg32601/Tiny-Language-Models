# configs/

This folder contains configuration and documentation files for each pretrained Tiny-Language-Model release.

## What’s Here

- Each `.yml` file documents a specific pretrained model release tag.
- These files include architecture details, training settings, and usage notes.

## How to Use

- Use the tag from the desired YAML file to load the matching model:

  ```python
  !git clone https://github.com/Rg32601/Tiny-Language-Models
  
  from configs.utils import download_and_load_pre_trained_model
  model = download_and_load_pre_trained_model("<model_tag>")
  ```
