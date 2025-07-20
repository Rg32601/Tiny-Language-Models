# Tiny Language Models

**This repository contains code accompanying our paper:**

> **[Tiny Language Models]**  
**Ronit D. Grossa¹, Yarden Tzacha¹, Tal Halevia, Ella Koresha, and Ido Kanter²,***  
¹ Department of Physics, Bar-Ilan University, Ramat-Gan, 52900, Israel  
² Gonda Interdisciplinary Brain Research Center, Bar-Ilan University, Ramat-Gan, 52900, Israel  
\* Corresponding author: [ido.kanter@biu.ac.il](mailto:ido.kanter@biu.ac.il)  
¹ These authors equally contributed to this work

---

If you find this repository useful, please cite:

```bibtex
@article{yourcitation2025,
    author = {Your Name and Co-authors},
    title = {Your Paper Title},
    journal = {Journal or Conference Name},
    year = {2025},
    volume = {XX},
    pages = {XX-XX},
    doi = {Your DOI here},
    url = {Paper URL}
}
 ```
This repository demonstrates pre-training and fine-tuning a compact BERT model (optionally with convolutional layers) using Masked Language Modeling (MLM) on Wikipedia, then fine-tuning for text classification tasks like AG News or DBpedia. It’s modular and PyTorch-based, utilizing Hugging Face Transformers.

## Abstract
A prominent achievement of natural language processing (NLP) is its ability to understand and generate meaningful human language. This capability relies on complex feedforward transformer block architectures pre-trained on large language models (LLMs). However, LLM pre-training is currently feasible only for a few dominant companies due to the immense computational resources required, limiting broader research participation. This creates a critical need for more accessible alternatives. In this study, we explore whether tiny language models (TLMs) exhibit the same key qualitative features of LLMs. We demonstrate that TLMs exhibit a clear performance gap between pre-trained and non-pre-trained models across classification tasks, indicating the effectiveness of pre-training, even at a tiny scale. The performance gap increases with the size of the pre-training dataset and with greater overlap between tokens in the pre-training and classification datasets. Furthermore, the classification accuracy achieved by a pre-trained deep TLM architecture can be replicated through a soft committee of multiple, independently pre-trained shallow architectures, enabling low-latency TLMs without affecting classification accuracy. Our results are based on pre-training BERT-6 and variants of BERT-1 on subsets of the Wikipedia dataset and evaluating their performance on FewRel, AGNews, and DBPedia classification tasks. Future research on TLM is expected to further illuminate the mechanisms underlying NLP, especially given that its biologically inspired models suggest that TLMs may be sufficient for children or adolescents to develop language. 

# How to run

## Features

- Pre-train compact BERT with optional convolutional layers
- Fine-tune for text classification
- Dataset loading, sampling, and tokenization utilities
- Modular and easy-to-extend structure

## Installation

```bash
pip install torch datasets transformers tqdm
```

## Usage

**Pre-train BERT6 on Wikipedia**

```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

args = init_parser('agnews')
wiki_dataloader = load_wikipedia_subset(args.pre_train_size)
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
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=epochs // 100, num_training_steps=epochs)

for epoch in range(epochs):
    mlm_train(wiki_dataloader, pre_train_model, optimizer)
    scheduler.step()
```

**Fine-tune for Classification (e.g., AG News)**

```python
train_dataloader, test_dataloader = load_agnews_dataloaders(args.samples_per_label_train, args.samples_per_label_test)

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

model.load_state_dict(pre_train_model.state_dict(), strict=False)
model.to(device)

optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_epochs)

for epoch in range(num_epochs):
    cls_train(train_dataloader, optimizer, criterion, model)
    scheduler.step()
    acc = cls_test(test_dataloader, model)
    print(f"Test Accuracy: {acc:.4f}")
```

## Customization

- To switch datasets, modify: `init_parser('dbpedia')` or `init_parser('fewrel')`.
- Adjust model and training parameters directly in the scripts as desired.

## Credits

Built using [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://huggingface.co/transformers/).
