from transformers import BertConfig, BertForMaskedLM
from nlp_bert_conv.models import build_bert_with_optional_conv_for_pre_train
import os
import requests
import re

def parse_model_name(model_name):
    """
    Parses a TinyBERT model name of the form:
    pretrain-tinybert-layers6-conv0-k3-d3-maxlen128-vocab30522-ep50-wikisize90000
    and returns a dictionary of its configuration values.
    """
    pattern = (
        r"Tiny-Language-Models/models/pretrain-tinybert"
        r"-layers(?P<bert_layers>\d+)"
        r"-conv(?P<conv_layers>\d+)"
        r"-k(?P<kernel>\d+)"
        r"-d(?P<d>\d+)"
        r"-heads(?P<num_attention_heads>\d+)"
        r"-maxlen(?P<max_length>\d+)"
        r"-vocab(?P<vocab_size>\d+)"
        r"-ep(?P<pre_train_epochs>\d+)"
        r"-wikisize(?P<pre_train_size>\d+)"
    )
    match = re.match(pattern, model_name)
    if not match:
        raise ValueError(f"Invalid model name format: {model_name}")
    # Convert all values to int
    config = {k: int(v) for k, v in match.groupdict().items()}
    return config
    
def load_pretrained_model(model_dir):
    """
    Loads the pretrained TinyBERT model from the given directory.

    Args:
        model_dir (str): Path to the model directory with config.json and model.safetensors

    Returns:
        Model loaded and ready to use.
    """
    config = BertConfig.from_pretrained(model_dir)
    params = parse_model_name(model_dir)
    model = build_bert_with_optional_conv_for_pre_train(
        hidden_size=  params["num_attention_heads"]*64,
        num_hidden_layers=params["bert_layers"],
        num_conv_layers=params["conv_layers"],
        kernel_size=params["kernel"],
        num_attention_heads= params["num_attention_heads"],
        intermediate_size= params["num_attention_heads"]*64*4,
        max_position_embeddings= 128,
        conv_channels_dim=params["d"],
        vocab_size= 30522,
        cls_token_id=0,
        pad_token_id=101,
        sep_token_id=102,
        mask_token_id=103,
    )

    # This will automatically use model.safetensors if present
    model = model.from_pretrained(model_dir, config=config)
    return model


def download_and_load_pre_trained_model(
    model_tag,
    repo="Rg32601/Tiny-Language-Models"
):
    """
    Downloads (if needed) and loads the pretrained TinyBERT model.

    Args:
        model_tag (str): The release tag name.
        repo (str): The github repo as "username/repo".

    Returns:
        Model loaded and ready to use.
    """
    model_dir = f"Tiny-Language-Models/models/{model_tag}"
    os.makedirs(model_dir, exist_ok=True)
    base_url = f"https://github.com/{repo}/releases/download/{model_tag}/"

    for fname in ["config.json", "model.safetensors"]:
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname} from GitHub release...")
            url = base_url + fname
            r = requests.get(url)
            if r.status_code != 200:
                raise RuntimeError(f"Failed to download {url} (status {r.status_code})")
            with open(fpath, "wb") as f:
                f.write(r.content)
    print("All model files downloaded.")
    return load_pretrained_model(model_dir)
