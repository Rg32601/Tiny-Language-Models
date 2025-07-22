from transformers import BertConfig, BertForMaskedLM
from nlp_bert_conv.models import build_bert_with_optional_conv_for_pre_train
import os
import requests

def load_pretrained_model(model_dir):
    """
    Loads the pretrained TinyBERT model from the given directory.

    Args:
        model_dir (str): Path to the model directory with config.json and model.safetensors

    Returns:
        Model loaded and ready to use.
    """
    config = BertConfig.from_pretrained(model_dir)
    # You can adjust these params to match your model; or make them dynamic if stored in config
    model = build_bert_with_optional_conv_for_pre_train(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers - 1,  # adjust if needed
        num_conv_layers=0,                               # adjust if needed
        kernel_size=3,                                   # adjust if needed
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        conv_channels_dim=3,                             # adjust if needed
        vocab_size=config.vocab_size,
        cls_token_id=config.cls_token_id,
        pad_token_id=config.pad_token_id,
        sep_token_id=config.sep_token_id,
        mask_token_id=config.mask_token_id,
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
    return load_pretrained_tinybert(model_dir)
