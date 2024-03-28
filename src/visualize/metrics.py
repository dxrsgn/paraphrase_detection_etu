import mlflow
from matplotlib import pyplot as plt

def get_metric(tracking_uri, run_id, metric_name):
    client  = mlflow.tracking.MlflowClient(
        tracking_uri=tracking_uri
    )
    metric = client.get_metric_history(run_id, metric_name)
    return [x.value for x in metric]

def plot_loss(tracking_uri, run_id):
    train_loss = get_metric(tracking_uri, run_id, "train_loss")
    val_loss = get_metric(tracking_uri, run_id, "val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_loss, label = "train loss")
    plt.plot(val_loss, label = "val loss")
    plt.legend()

def plot_metric(tracking_uri, run_id, metric_name):
    metric = get_metric(tracking_uri, run_id, metric_name)
    if len(metric) == 0:
        print(f"There is no metric {metric_name}")
        return
    plt.title(metric_name)
    plt.xlabel("Step")
    plt.plot(metric, label = metric_name)
    plt.legend()
