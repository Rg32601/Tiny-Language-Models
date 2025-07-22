## 🔥 How to Load the Custom Pretrained Model

1. **Clone this repository:**
    ```bash
    git clone https://github.com/<your-username>/<your-repo>.git
    cd <your-repo>
    ```

2. **Download and unzip the model from [Releases](https://github.com/<your-username>/<your-repo>/releases).**

3. **Edit and run `main.py`** to load the model:
    ```python
    import torch
    from your_model_file import build_bert_with_optional_conv_for_pre_train

    model = build_bert_with_optional_conv_for_pre_train(
        hidden_size=384,          # Example: fill with your values
        num_hidden_layers=6,
        num_conv_layers=2,
        kernel_size=3,
        num_attention_heads=6,
        intermediate_size=1536,
        max_position_embeddings=128,
        conv_channels_dim=64,
        vocab_size=30522,
        cls_token_id=0,
        pad_token_id=101,
        sep_token_id=102,
        mask_token_id=103,
    )
    state_dict = torch.load("pretrain-tinybert-layers6-conv2-k3-d64-maxlen128-vocab30522-ep10-lr0.0001-wikisize100000/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
    ```
    Replace parameters as needed.  
    **Don’t use `BertForMaskedLM.from_pretrained()`—it will not load the convolutional layers.**
