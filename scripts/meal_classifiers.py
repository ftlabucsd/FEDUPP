import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # Tensor of shape (num_samples, seq_len)
        self.y = y  # Tensor of shape (num_samples,)

    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
class RNNClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, num_classes=2):
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


def train(model:nn.Module, lr:float, num_epochs:int, train_loader:DataLoader, 
          X_test_tensor:torch.tensor, y_test_tensor:torch.tensor):
    
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