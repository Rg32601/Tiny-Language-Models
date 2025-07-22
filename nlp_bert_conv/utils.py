import torch
import argparse

def mask_tokens(
    inputs: torch.Tensor,
    mlm_probability: float = 0.15,
    vocab_size: int = 30522,
    pad_token_id: int = 0,
    cls_token_id: int = 101,
    sep_token_id: int = 102,
    mask_token_id: int = 103
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling (MLM) as in BERT.

    Args:
        inputs (torch.Tensor): Input token IDs (batch_size x seq_len).
        mlm_probability (float): Probability of masking each token.
        vocab_size (int): Size of the tokenizer vocabulary.
        pad_token_id (int): Token ID used for padding.
        cls_token_id (int): Token ID for [CLS] token.
        sep_token_id (int): Token ID for [SEP] token.
        mask_token_id (int): Token ID for [MASK] token.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of (masked_input_ids, labels),
            where labels have -100 for non-MLM positions (ignored in loss).
    """
    labels = inputs.clone()

    # Create mask for special tokens
    special_tokens_mask = (inputs == pad_token_id) | (inputs == cls_token_id) | (inputs == sep_token_id)
    
    # Decide which tokens to mask
    probability_matrix = torch.full(labels.shape, mlm_probability)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Labels: only compute loss on masked tokens
    labels[~masked_indices] = -100

    # 80% of the time, replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_token_id

    # 10% of the time, replace with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # 10% of the time, keep the original token (no need to modify)

    return inputs, labels


def init_parser(dataset):
    """
    Initialize argument parser with dataset-specific defaults.

    Parameters:
        dataset (str): One of {"agnews", "dbpedia", "fewrel"}.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    allowed_datasets = {"agnews", "dbpedia", "fewrel"}
    if dataset not in allowed_datasets:
        raise ValueError(f"Invalid dataset '{dataset}'. Must be one of: {', '.join(allowed_datasets)}.")

    # Dataset-specific default values
    dataset_defaults = {
        "agnews": {
            "samples_per_label_train": 1000,
            "samples_per_label_test": 1000,
            "num_labels": 4
        },
        "dbpedia": {
            "samples_per_label_train": 100,
            "samples_per_label_test": 100,
            "num_labels": 14
        },
        "fewrel": {
            "samples_per_label_train": 630,
            "samples_per_label_test": 70,
            "num_labels": 64
        }
    }

    defaults = dataset_defaults[dataset]

    parser = argparse.ArgumentParser(description=f"Configure BERT6 pretraining on the {dataset.upper()} dataset.")

    # General training settings
    parser.add_argument("--epochs", type=int, default=50, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay during fine-tuning")

    # Pre-training settings
    parser.add_argument("--pre_train_epochs", type=int, default=50, help="Number of pre-training epochs")
    parser.add_argument("--pre_train_lr", type=float, default=5e-5, help="Learning rate for pre-training")
    parser.add_argument("--pre_train_weight_decay", type=float, default=1e-2, help="Weight decay during pre-training")
    parser.add_argument("--pre_train_size", type=int, default=40000, help="Number of samples used for pre-training")

    # Data and batching
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--samples_per_label_train", type=int, default=defaults["samples_per_label_train"],
                        help="Train samples per label")
    parser.add_argument("--samples_per_label_test", type=int, default=defaults["samples_per_label_test"],
                        help="Test samples per label")

    # Masked language modeling
    parser.add_argument("--mlm_prob", type=float, default=0.15, help="Probability of masking tokens for MLM")

    # Model architecture
    parser.add_argument("--bert_layers", type=int, default=6, help="Number of BERT encoder layers")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of attention heads per layer")
    parser.add_argument("--head_size", type=int, default=64, help="Dimention of each attention head")
    parser.add_argument("--conv_layers", type=int, default=0, help="Number of convolutional layers (if any)")
    parser.add_argument("--kernel", type=int, default=3, help="Kernel size for convolutional layers")
    parser.add_argument("--d", type=int, default=3, help="Number of channels for convolutional layers")

    # Tokenizer and vocabulary
    parser.add_argument("--vocab_size", type=int, default=30522, help="Size of tokenizer vocabulary")
    parser.add_argument("--pad_token_id", type=int, default=0, help="ID for [PAD] token")
    parser.add_argument("--cls_token_id", type=int, default=101, help="ID for [CLS] token")
    parser.add_argument("--sep_token_id", type=int, default=102, help="ID for [SEP] token")
    parser.add_argument("--mask_token_id", type=int, default=103, help="ID for [MASK] token")

    # Misc
    parser.add_argument("--num_labels", type=int, default=defaults["num_labels"], help="Number of classification labels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_known_args()[0]
