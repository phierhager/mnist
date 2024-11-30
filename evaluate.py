import torch


import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    softmax: bool = True,
):
    # Set the model to evaluation mode
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            # Get the model's predictions
            outputs = model(images)

            # Apply softmax if needed (for multi-class classification)
            if softmax:
                outputs = torch.softmax(outputs, dim=1)

            # Get the predicted class
            _, predicted = torch.max(outputs, 1)

            # Update the total number of samples and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store the true labels and predicted labels for further metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Compute precision, recall, and F1-score (macro averaged for multi-class)
    precision = precision_score(all_labels, all_predictions, average="macro")
    recall = recall_score(all_labels, all_predictions, average="macro")
    f1 = f1_score(all_labels, all_predictions, average="macro")

    # If it's a binary classification task, you can extract these values from the confusion matrix
    if cm.shape == (2, 2):  # Binary classification case
        tn, fp, fn, tp = cm.ravel()
        false_positives = fp
        false_negatives = fn
    else:
        false_positives = None
        false_negatives = None

    # Print the results
    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if false_positives is not None and false_negatives is not None:
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")

    # Return the metrics
    return accuracy, precision, recall, f1, false_positives, false_negatives
