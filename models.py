from torch import nn
import torch
import numpy as np


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.fc(x)


def pseudo_inverse(x: torch.Tensor):
    """Compute the pseudo-inverse of the input matrix.

    The input matrix should be a square matrix.
    """
    print((x.T @ x).shape)
    print(torch.inverse(x.T @ x).shape)
    return torch.inverse(x.T @ x) @ x.T


class LSClassifier(nn.Module):
    """Using classification direct with least squares loss.

    For more information, see Bishop, 2024, p. 136-137.
    """

    def __init__(self, imgs_flattened: torch.Tensor, labels: torch.Tensor):
        """Using classification direct with least squares loss.

        For more information, see Bishop, 2024, p. 136-137.
        The input should be all the training data.
        The first dimension should be the number of training samples.

        Args:
            imgs_flattened (torch.Tensor): Flattened images.
                (N, width * height* channels)
            labels (torch.Tensor): Labels. (N, num_classes)
        """
        super().__init__()
        imgs_flattened = torch.cat(
            (torch.ones((imgs_flattened.shape[0], 1)), imgs_flattened), dim=1
        )
        self.weight_matrix: torch.Tensor = (
            torch.linalg.pinv(imgs_flattened) @ labels
        )

    def forward(self, x: torch.Tensor):
        """The output is the class label. And there are no weights to learn."""
        with torch.no_grad():
            x_tilde = torch.cat((torch.ones((x.shape[0], 1)), x), dim=1).T
            return (self.weight_matrix.T @ x_tilde).T


def gaussian_pdf(x, mean, covariance):
    d = len(x)
    diff = x - mean
    exponent = -0.5 * torch.dot(
        torch.dot(diff.T, torch.inverse(covariance)), diff
    )
    norm_const = 1 / torch.sqrt((2 * np.pi) ** d * torch.det(covariance))
    return norm_const * torch.exp(exponent)


class GaussianMixtureModel(nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()
        self.coefficients = [1 / num_classes] * num_classes
        self.means = nn.Parameter(torch.randn(num_classes, num_features))
        self.covariances = nn.Parameter(
            torch.randn(num_classes, num_features, num_features)
        )

    def get_probability(self, x):
        probabilities = []
        for i in range(len(self.coefficients)):
            probabilities.append(
                self.coefficients[i]
                * gaussian_pdf(x, self.means[i], self.covariances[i])
            )
        return torch.sum(probabilities, dim=0)

    def get_responsibilities(self, x):
        probabilities = []
        for i in range(len(self.coefficients)):
            probabilities.append(
                self.coefficients[i]
                * gaussian_pdf(x, self.means[i], self.covariances[i])
            )
        return probabilities / torch.sum(probabilities, dim=0)

    def expectation_step(self, x):
        responsibilities = self.get_responsibilities(x)
        return responsibilities

    def maximization_step(self, x, responsibilities):
        N = x.shape[0]
        for i in range(len(self.coefficients)):
            N_k = torch.sum(responsibilities[i])
            self.coefficients[i] = N_k / N
            self.means[i] = (
                torch.sum(responsibilities[i].unsqueeze(1) * x, dim=0) / N_k
            )
            diff = x - self.means[i]
            self.covariances[i] = (
                torch.sum(
                    responsibilities[i].unsqueeze(1).unsqueeze(2)
                    * diff.unsqueeze(1)
                    * diff.unsqueeze(2),
                    dim=0,
                )
                / N_k
            )

    def log_likelihood(self, x):

        return probabilities / torch.sum(probabilities, dim=0)
