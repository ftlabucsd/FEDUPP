"""
This script defines and implements neural network models (RNN and CNN) for classifying
meal-related time-series data. It includes classes for datasets, the classifiers
themselves, and functions for training, evaluation, and prediction.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TimeSeriesDataset(Dataset):
    """A custom PyTorch Dataset for time-series data."""
    def __init__(self, X, y):
        """
        Args:
            X (torch.Tensor): The input features, with shape (num_samples, seq_len).
            y (torch.Tensor): The corresponding labels, with shape (num_samples,).
        """
        self.X = X  # Tensor of shape (num_samples, seq_len)
        self.y = y  # Tensor of shape (num_samples,)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.y)
        
    def __getitem__(self, idx):
        """Returns a single sample and its label."""
        return self.X[idx], self.y[idx]
    
    
class RNNClassifier(nn.Module):
    """A Recurrent Neural Network (RNN) classifier using LSTM layers."""
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, num_classes=2):
        """
        Args:
            input_size (int): The number of features in the input.
            hidden_size (int): The number of features in the hidden state.
            num_layers (int): The number of recurrent layers.
            num_classes (int): The number of output classes.
        """
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = x.unsqueeze(-1)  # Now x is [batch_size, seq_len, 1]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class CNNClassifier(nn.Module):
    """A 1D Convolutional Neural Network (CNN) for time-series classification."""
    def __init__(self, num_classes=2, maxlen=4):
        """
        Args:
            num_classes (int): The number of output classes.
            maxlen (int): The maximum length of the input sequences.
        """
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.maxlen = maxlen
        # Calculate the size of the feature map after convolution and pooling
        self.feature_size = self._get_feature_size()
        self.fc = nn.Linear(self.feature_size, num_classes)
        
    def _get_feature_size(self):
        # Replace 'maxlen' with your actual maximum sequence length
        dummy_input = torch.zeros(1, 1, self.maxlen)  # Shape: [batch_size, channels, seq_len]
        x = self.relu(self.conv1(dummy_input))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        feature_size = x.view(1, -1).size(1)
        return feature_size
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = x.unsqueeze(1)  # Shape: [batch_size, channels=1, seq_len]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


def train(model:nn.Module, lr:float, num_epochs:int, train_loader:DataLoader, 
          X_test_tensor:torch.tensor, y_test_tensor:torch.tensor):
    """
    Trains a given neural network model.

    Args:
        model (nn.Module): The model to be trained.
        lr (float): The learning rate for the optimizer.
        num_epochs (int): The number of training epochs.
        train_loader (DataLoader): The DataLoader for the training data.
        X_test_tensor (torch.Tensor): The test features.
        y_test_tensor (torch.Tensor): The test labels.

    Returns:
        nn.Module: The trained model.
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += y_batch.size(0)
            predicted, y_batch = predicted.cpu().numpy(), y_batch.cpu().numpy()
            correct_train += (predicted == y_batch).sum().item()

        avg_loss = total_loss / total_train
        train_accuracy = correct_train / total_train

        model.eval()
        with torch.no_grad():
            outputs_test = model(X_test_tensor)
            _, predicted_test = torch.max(outputs_test.data, 1)
            correct_test = (predicted_test.cpu() == y_test_tensor).sum().item()
            test_accuracy = correct_test / y_test_tensor.size(0)

        if (epoch+1) % 10 == 0 or epoch == num_epochs-1:
            print(f'Ep {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
    
    return model


def evaluate_meals_by_groups(model:nn.Module, ctrl_input:torch.Tensor, ctrl_y:torch.Tensor,
                             exp_input:torch.Tensor, exp_y:torch.Tensor):
    """
    Evaluates the model's performance on control and experimental groups.

    Args:
        model (nn.Module): The trained model.
        ctrl_input (torch.Tensor): Input data for the control group.
        ctrl_y (torch.Tensor): Labels for the control group.
        exp_input (torch.Tensor): Input data for the experimental group.
        exp_y (torch.Tensor): Labels for the experimental group.
    """
    model.eval()
    with torch.no_grad():
        outputs_ctrl = model(ctrl_input)
        _, predicted_ctrl = torch.max(outputs_ctrl.data, 1)
        correct_test = (predicted_ctrl.cpu().numpy() == ctrl_y).sum().item()
        ctrl_accuracy = correct_test / len(ctrl_y)

        outputs_exp = model(exp_input)
        _, predicted_exp = torch.max(outputs_exp.data, 1)
        correct_test = (predicted_exp.cpu().numpy() == exp_y).sum().item()
        exp_accuracy = correct_test / len(exp_y)

    print(f'Control Accuracy: {ctrl_accuracy:.3f}, Exp Accuracy: {exp_accuracy:.3f}')

    predicted_ctrl, predicted_exp = predicted_ctrl.cpu().numpy(), predicted_exp.cpu().numpy()
    ctrl_good, ctrl_total = np.sum(predicted_ctrl), np.size(predicted_ctrl)
    exp_good, exp_total = np.sum(predicted_exp), np.size(predicted_exp)

    print(f'Control Group: {ctrl_total-ctrl_good}/{ctrl_total} good meals with proportion of {1-ctrl_good/ctrl_total}')
    print(f'Experiment Group: {exp_total-exp_good}/{exp_total} good meals with proportion of {1-exp_good/exp_total}')


def evaluate_meals_on_new_data(model:nn.Module, ctrl_input:torch.Tensor, exp_input:torch.Tensor):
    """
    Evaluates the model on new, unlabeled data from control and experimental groups.

    Args:
        model (nn.Module): The trained model.
        ctrl_input (torch.Tensor): New input data for the control group.
        exp_input (torch.Tensor): New input data for the experimental group.
    """
    ctrl_input, exp_input = ctrl_input.to(device), exp_input.to(device)
    model.eval()
    with torch.no_grad():
        outputs_ctrl = model(ctrl_input)
        _, predicted_ctrl = torch.max(outputs_ctrl.data, 1)

        outputs_exp = model(exp_input)
        _, predicted_exp = torch.max(outputs_exp.data, 1)

    predicted_ctrl, predicted_exp = predicted_ctrl.cpu().numpy(), predicted_exp.cpu().numpy()
    ctrl_good, ctrl_total = np.sum(predicted_ctrl), np.size(predicted_ctrl)
    exp_good, exp_total = np.sum(predicted_exp), np.size(predicted_exp)

    print(f'Control Group: {ctrl_total-ctrl_good}/{ctrl_total} good meals with proportion of {1-ctrl_good/ctrl_total}')
    print(f'Experiment Group: {exp_total-exp_good}/{exp_total} good meals with proportion of {1-exp_good/exp_total}')



def predict(model:nn.Module, input):
    """
    Makes predictions on new data using the trained model.

    Args:
        model (nn.Module): The trained model.
        input (array-like or torch.Tensor): The input data for prediction.

    Returns:
        np.ndarray: The predicted class labels.
    """
    if type(input) != torch.Tensor: input = torch.Tensor(input)
    input = input.to(device)

    model.eval()
    with torch.no_grad():
        outputs_ctrl = model(input)
        _, predicted_ctrl = torch.max(outputs_ctrl.data, 1)

    return predicted_ctrl.cpu().numpy()

def count_parameters(model:nn.Module):
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (nn.Module): The model to inspect.
    """
    trainable_params = sum(p.numel() for p in model.parameters())
    print(f'Trainable parameters: {trainable_params}')