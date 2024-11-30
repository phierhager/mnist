from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train(
    epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    plot_loss: bool = True,
):
    loss_history = []  # To store loss values

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0  # Accumulate loss for the epoch

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    if plot_loss:
        plot_loss_history(loss_history, epochs)


def plot_loss_history(loss_history, epochs):
    # Plot the loss history
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), loss_history, marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()
