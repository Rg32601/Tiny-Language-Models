import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import mask_tokens

def cls_test(
    test_dataloader: DataLoader,
    model: nn.Module
) -> float:
    """
    Evaluate a classification model with accuracy metric.

    Args:
        test_dataloader (DataLoader): Test set loader.
        model (nn.Module): Model that outputs logits.

    Returns:
        float: Accuracy score on the test set.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    correct = 0
    total = 0

    progress_bar = tqdm(test_dataloader, desc="Testing", leave=True)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.float)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        progress_bar.set_postfix(acc=f"{accuracy:.4f}")

    return accuracy

def cls_train(
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    model: nn.Module
) -> None:
    """
    Train a classification model using mixed precision and display progress.

    Args:
        train_dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer for model parameters.
        criterion (nn.Module): Loss function.
        model (nn.Module): Model that outputs logits.
    """
    scaler = torch.cuda.amp.GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)

    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.float)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def mlm_train(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Trains a masked language model for one epoch with a tqdm progress bar showing live loss.

    Args:
        dataloader (DataLoader): Unlabeled dataset loader for MLM pretraining.
        model (nn.Module): Model with MLM objective (e.g., BertForMaskedLM).
        optimizer (Optimizer): Optimizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler()

    progress_bar = tqdm(dataloader, desc="MLM Training", leave=True)

    for batch in progress_bar:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch.get('token_type_ids', None)

        masked_input_ids, mlm_labels = mask_tokens(input_ids.clone())

        masked_input_ids = masked_input_ids.to(device)
        attention_mask = attention_mask.to(device)
        mlm_labels = mlm_labels.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=mlm_labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
