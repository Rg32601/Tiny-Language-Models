from torch import nn
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
)

class ConvolutionLayers(nn.Module):
    """
    A modular convolutional block used to optionally replace the first layer
    of a BERT encoder. Adds spatial context through stacked 2D convolutions.
    """

    def __init__(self, num_layers, channels, kernel_size):
        """
        Initialize a stack of convolutional layers.

        Args:
            num_layers (int): Number of convolutional blocks.
            channels (List[int]): List of channel dimensions. Length = num_layers + 1.
            kernel_size (int): Kernel size for each convolution.
        """
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, padding="same"),
                nn.BatchNorm2d(channels[i + 1]),
                nn.GELU()
            ])

        # Final projection to 1 channel
        self.conv = nn.Sequential(
            nn.Conv2d(channels[-1], 1, kernel_size=1, padding="same"),
            nn.BatchNorm2d(1),
            nn.GELU()
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=None,
    ):
        """
        Forward pass through convolutional layers.

        Args:
            hidden_states (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).

        Returns:
            Tuple[Tensor, Optional[Tensor]]:
                - Processed hidden states: (batch_size, seq_len, hidden_dim)
                - attention_mask (unchanged)
        """
        x = hidden_states.view(hidden_states.size(0), 1, hidden_states.size(1), hidden_states.size(2))

        for layer in self.layers:
            x = layer(x)

        x = self.conv(x)
        x = x.view(x.size(0), x.size(2), x.size(3))

        return x, attention_mask


def build_bert_with_optional_conv_for_pre_train(
    hidden_size,
    num_hidden_layers,
    num_conv_layers,
    kernel_size,
    num_attention_heads,
    intermediate_size,
    max_position_embeddings,
    conv_channels_dim,
    vocab_size=30522,
    cls_token_id=0,
    pad_token_id=101,
    sep_token_id=102,
    mask_token_id=103,
):
    """
    Construct a BERT model for masked language modeling with optional convolutional preprocessing.

    Args:
        hidden_size (int): Transformer hidden size.
        num_hidden_layers (int): Number of BERT transformer layers (not including CNN).
        num_conv_layers (int): If > 0, replaces first encoder layer with CNN block.
        kernel_size (int): Kernel size for convolutional layers.
        num_attention_heads (int): Number of self-attention heads per layer.
        intermediate_size (int): Size of the feedforward layer.
        max_position_embeddings (int): Max sequence length for positional embeddings.
        conv_channels_dim (int): Output channels for each convolutional layer.
        vocab_size (int): Token vocabulary size.
        cls_token_id, pad_token_id, sep_token_id, mask_token_id (int): Special token IDs.

    Returns:
        BertForMaskedLM: Configured model.
    """
    config = BertConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers + int(num_conv_layers > 0),
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
        cls_token_id=cls_token_id,
        pad_token_id=pad_token_id,
        sep_token_id=sep_token_id,
        mask_token_id=mask_token_id,
    )

    model = BertForMaskedLM(config)

    if num_conv_layers > 0:
        conv_block = ConvolutionLayers(
            num_layers=num_conv_layers,
            channels=[1] + [conv_channels_dim] * num_conv_layers,
            kernel_size=kernel_size
        )
        model.bert.encoder.layer[0] = conv_block

    return model


def build_bert_with_optional_conv_for_classification(
    hidden_size,
    num_hidden_layers,
    num_conv_layers,
    kernel_size,
    num_attention_heads,
    intermediate_size,
    max_position_embeddings,
    conv_channels_dim,
    num_labels,
    vocab_size=30522,
    cls_token_id=0,
    pad_token_id=101,
    sep_token_id=102,
    mask_token_id=103,
):
    """
    Construct a BERT model for sequence classification with optional convolutional preprocessing.

    Args:
        hidden_size (int): Transformer hidden size.
        num_hidden_layers (int): Number of BERT transformer layers (not including CNN).
        num_conv_layers (int): If > 0, replaces first encoder layer with CNN block.
        kernel_size (int): Kernel size for convolutional layers.
        num_attention_heads (int): Number of self-attention heads per layer.
        intermediate_size (int): Size of the feedforward layer.
        max_position_embeddings (int): Max sequence length for positional embeddings.
        conv_channels_dim (int): Output channels for each convolutional layer.
        num_labels (int): Number of output classes for classification.
        vocab_size (int): Token vocabulary size.
        cls_token_id, pad_token_id, sep_token_id, mask_token_id (int): Special token IDs.

    Returns:
        BertForSequenceClassification: Configured model.
    """
    config = BertConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers + int(num_conv_layers > 0),
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
        num_labels=num_labels,
        cls_token_id=cls_token_id,
        pad_token_id=pad_token_id,
        sep_token_id=sep_token_id,
        mask_token_id=mask_token_id,
    )

    model = BertForSequenceClassification(config)

    if num_conv_layers > 0:
        conv_block = ConvolutionLayers(
            num_layers=num_conv_layers,
            channels=[1] + [conv_channels_dim] * num_conv_layers,
            kernel_size=kernel_size
        )
        model.bert.encoder.layer[0] = conv_block

    return model
