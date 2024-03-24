import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, F1Score
import time
import mlflow

def train_model(model, criterion, optimizer, dataloaders, modeltype, num_epochs=10):
    since = time.time()
    train_batches = len(dataloaders["train"])
    val_batches = len(dataloaders["val"])
    metrics = {
        "Accuracy":  Accuracy(task = "binary").to("cuda"),
        "Precision": Precision(task = "binary").to("cuda"),
        "Recall": Recall(task = "binary").to("cuda"),
        "F1Score": F1Score(task = "binary").to("cuda")
    }
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        loss_train = 0
        loss_val = 0
        model.train()
        for i, data in enumerate(dataloaders["train"]):
            print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)
            inputs = data["input_ids"]
            labels = data["labels"]
            inputs, labels = inputs.cuda(), labels.cuda()
            if modeltype == "transformer":
                mask = data["attention_mask"].cuda()
                outputs = model(inputs, mask)
            else:
                outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            for metric in metrics.values():
                metric.update(outputs, labels)
        mlflow.log_metric(f"train_loss", loss_train, step=epoch)
        for metric_name, metric in metrics.items():
            mlflow.log_metric(f"train_{metric_name}", metric.compute().item(), step=epoch)
            # Probably explicit reseting is not neccessary
            metric.reset()
        model.eval()
        print()
        for i, data in enumerate(dataloaders["val"]):
            print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
            inputs = data["input_ids"]
            labels = data["labels"]
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            if modeltype == "transformer":
                mask = data["attention_mask"].cuda()
                outputs = model(inputs, mask)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss_val += loss.item()
            for metric in metrics.values():
                metric.update(outputs, labels)
        mlflow.log_metric(f"val_loss", loss_val, step=epoch)
        for metric_name, metric in metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric.compute().item(), step=epoch)
            # Probably explicit reseting is not neccessary
            metric.reset()
        print()
    elapsed_time = time.time() - since
    mlflow.log_metric("training time", elapsed_time)
    print("\nTraining completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    return model
