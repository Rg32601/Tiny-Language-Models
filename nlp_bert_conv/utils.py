import torch

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
