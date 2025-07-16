import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from nlp_bert_conv.data import load_agnews_dataloaders, load_wikipedia_subset
from nlp_bert_conv.models import (
    build_bert_with_optional_conv_for_pre_train,
    build_bert_with_optional_conv_for_classification
)
from nlp_bert_conv.train import mlm_train, cls_train, cls_test
from nlp_bert_conv.utils import init_parser

# Initialize argument parser and get configuration for AG News
args = init_parser('agnews')

# Prepare dataloaders for pre-training and fine-tuning
train_dataloader, test_dataloader = load_agnews_dataloaders(
    args.samples_per_label_train, args.samples_per_label_test
)
wiki_dataloader = load_wikipedia_subset(args.pre_train_size)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Pre-train BERT with (optional) Conv Layers on Wikipedia (MLM)
# -------------------------------
pre_train_model = build_bert_with_optional_conv_for_pre_train(
    hidden_size=args.head_size * args.num_attention_heads,
    num_hidden_layers=args.bert_layers,
    num_conv_layers=args.conv_layers,
    kernel_size=args.kernel,
    num_attention_heads=args.num_attention_heads,
    intermediate_size=args.head_size * args.num_attention_heads * 4,
    max_position_embeddings=args.max_length,
    conv_channels_dim=args.d,
    vocab_size=args.vocab_size,
    cls_token_id=args.cls_token_id,
    pad_token_id=args.pad_token_id,
    sep_token_id=args.sep_token_id,
    mask_token_id=args.mask_token_id,
)

optimizer = AdamW(pre_train_model.parameters(), lr=args.pre_train_lr, weight_decay=args.pre_train_weight_decay)
epochs = args.pre_train_epochs
num_warmup_steps = max(epochs // 100, 1)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=epochs,
)

for epoch in range(epochs):
    print(f"Pretrain Epoch {epoch + 1}/{epochs}")
    mlm_train(wiki_dataloader, pre_train_model, optimizer)
    scheduler.step()

# -------------------------------
# Fine-tune for Classification on AG News (or other dataset)
# -------------------------------
model = build_bert_with_optional_conv_for_classification(
    hidden_size=args.head_size * args.num_attention_heads,
    num_hidden_layers=args.bert_layers,
    num_conv_layers=args.conv_layers,
    kernel_size=args.kernel,
    num_attention_heads=args.num_attention_heads,
    intermediate_size=args.head_size * args.num_attention_heads * 4,
    max_position_embeddings=args.max_length,
    conv_channels_dim=args.d,
    num_labels=args.num_labels,
    vocab_size=args.vocab_size,
    cls_token_id=args.cls_token_id,
    pad_token_id=args.pad_token_id,
    sep_token_id=args.sep_token_id,
    mask_token_id=args.mask_token_id,
)

# Load the pre-trained weights (except classifier head)
model.load_state_dict(pre_train_model.state_dict(), strict=False)
model.to(device)

optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = CrossEntropyLoss()
num_epochs = args.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_epochs
)

for epoch in range(num_epochs):
    print(f"Finetune Epoch {epoch + 1}/{num_epochs}")
    cls_train(train_dataloader, optimizer, criterion, model)
    scheduler.step()
    acc = cls_test(test_dataloader, model)
    print(f"Test Accuracy: {acc:.4f}")
