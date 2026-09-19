import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def evaluate_model(model, loader, device, model_name="Model"):
    """
    Runs the model over a DataLoader, returns:
      - all true labels
      - all predicted labels
      - all softmax probabilities (for ROC-AUC)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plots a labelled confusion matrix and saves it."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs,
                          model_name, save_path=None):
    """Plots loss and accuracy curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label='Train', color='#00C2CB')
    ax1.plot(epochs, val_losses, label='Val', color='#F5A623')
    ax1.set_title(f'{model_name} — Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(epochs, train_accs, label='Train', color='#00C2CB')
    ax2.plot(epochs, val_accs, label='Val', color='#F5A623')
    ax2.set_title(f'{model_name} — Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def print_metrics(y_true, y_pred, y_probs, model_name):
    """Prints classification report and ROC-AUC."""
    print(f"\n{'=' * 50}")
    print(f"Results for: {model_name}")
    print(f"{'=' * 50}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    print(f"ROC-AUC (OvR): {auc:.4f}")
