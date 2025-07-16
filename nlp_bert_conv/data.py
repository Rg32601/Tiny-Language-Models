import random
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer

def sample_per_label(dataset, max_samples_per_label, seed=42):
    """
    Returns a balanced subset of the dataset with up to `max_samples_per_label` samples per label.

    Args:
        dataset (datasets.Dataset): The dataset to sample from.
        max_samples_per_label (int): Max samples for each label.
        seed (int): Random seed.

    Returns:
        datasets.Dataset: Subsampled dataset.
    """
    random.seed(seed)
    label_to_indices = {}
    for idx, example in enumerate(dataset):
        label = example["label"]
        label_to_indices.setdefault(label, []).append(idx)

    sampled_indices = []
    for indices in label_to_indices.values():
        if len(indices) >= max_samples_per_label:
            sampled = random.sample(indices, max_samples_per_label)
        else:
            sampled = indices
        sampled_indices.extend(sampled)
    return dataset.select(sampled_indices)

def tokenize_dataset(dataset, tokenizer, input_key="content", max_length=128):
    """
    Applies a HuggingFace tokenizer to the dataset.

    Args:
        dataset (datasets.Dataset): The input dataset.
        tokenizer (PreTrainedTokenizer): A HuggingFace tokenizer instance.
        input_key (str): Dataset field to tokenize.
        max_length (int): Max token length.

    Returns:
        datasets.Dataset: Tokenized dataset (torch format).
    """
    tokenized = dataset.map(
        lambda x: tokenizer(
            x[input_key],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
    )
    columns = ["input_ids", "attention_mask"]
    if "label" in dataset.column_names:
        tokenized = tokenized.rename_column("label", "labels")
        columns.append("labels")
    tokenized.set_format(type="torch", columns=columns)
    return tokenized

def get_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Wraps a HuggingFace dataset as a PyTorch DataLoader.

    Args:
        dataset (datasets.Dataset): Tokenized dataset.
        batch_size (int): Loader batch size.
        shuffle (bool): Shuffle data.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_dbpedia_dataloaders(
    samples_per_label_train,
    samples_per_label_test,
    batch_size=32,
    seed=42,
    max_length=128,
):
    """
    Loads, balances, tokenizes, and returns PyTorch DataLoaders for DBpedia.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = load_dataset("dbpedia_14", split="train")
    test_dataset = load_dataset("dbpedia_14", split="test")
    small_train = sample_per_label(train_dataset, samples_per_label_train, seed)
    small_test = sample_per_label(test_dataset, samples_per_label_test, seed)
    tokenized_train = tokenize_dataset(small_train, tokenizer, "content", max_length)
    tokenized_test = tokenize_dataset(small_test, tokenizer, "content", max_length)
    train_loader = get_dataloader(tokenized_train, batch_size, shuffle=True)
    test_loader = get_dataloader(tokenized_test, batch_size, shuffle=False)
    return train_loader, test_loader

def load_agnews_dataloaders(
    samples_per_label_train,
    samples_per_label_test,
    batch_size=32,
    seed=42,
    max_length=128,
):
    """
    Loads, balances, tokenizes, and returns PyTorch DataLoaders for AG News.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = load_dataset("ag_news", split="train")
    test_dataset = load_dataset("ag_news", split="test")
    small_train = sample_per_label(train_dataset, samples_per_label_train, seed)
    small_test = sample_per_label(test_dataset, samples_per_label_test, seed)
    tokenized_train = tokenize_dataset(small_train, tokenizer, "text", max_length)
    tokenized_test = tokenize_dataset(small_test, tokenizer, "text", max_length)
    train_loader = get_dataloader(tokenized_train, batch_size, shuffle=True)
    test_loader = get_dataloader(tokenized_test, batch_size, shuffle=False)
    return train_loader, test_loader

def load_wikipedia_dataloader(
    pre_train_size=40000,
    seed=42,
    max_length=128,
    batch_size=32,
):
    """
    Loads and returns a PyTorch DataLoader for a random subset of the Wikipedia dataset.

    Returns:
        DataLoader: DataLoader for tokenized Wikipedia subset.
    """
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    wiki = wiki.shuffle(seed=seed)
    wiki_small = wiki.select(range(pre_train_size))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized = tokenize_dataset(wiki_small, tokenizer, input_key="text", max_length=max_length)
    dataloader = get_dataloader(tokenized, batch_size=batch_size, shuffle=True)
    return dataloader
