import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, F1Score
import time
import mlflow

def eval_model(model: nn.Module, dataloaders: list[DataLoader], modeltype: str):
    """Evaluate model on test set. Metrics: Accuracy, Precision, Recall, F1Score

    Args:
        model (nn.Module): Model
        dataloaders (list[DataLoader]): Dataloaders 
        modeltype (str): Model type ("lstm", "transformer") 
    """

    model.eval()
    since = time.time()
    metrics = {
        "Accuracy":  Accuracy(task = "binary", deivce = "cuda"),
        "Precision": Precision(task = "binary", deivce = "cuda"),
        "Recall": Recall(task = "binary", deivce = "cuda"),
        "F1Score": F1Score(task = "binary", deivce = "cuda")
    }
    print()
    for i, data in enumerate(dataloaders["test"]):
        inputs = data["input_ids"]
        labels = data["labels"]
        inputs, labels = inputs.cuda(), labels.cuda()
        if modeltype == "transformer":
            mask = data["attention_mask"].cuda()
            outputs = model(inputs, mask)
        else:
            outputs = model(inputs)
        for metric in metrics.values():
            metric.update(outputs, labels)
    for metric_name, metric in metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric.compute().item())
    print()
    elapsed_time = time.time() - since
    mlflow.log_metric("testing time", elapsed_time)
    print("\Testing completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
