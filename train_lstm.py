# train_lstm.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class FatigueSequenceDataset(Dataset):
    def __init__(self, data_path):
        self.sequences = []
        self.labels = []
        self.label_map = {'alert': 0, 'drowsy': 1}
        
        for label, numeric_label in self.label_map.items():
            folder_path = os.path.join(data_path, label)
            for filename in os.listdir(folder_path):
                if filename.endswith('.npy'):
                    sequence = np.load(os.path.join(folder_path, filename))
                    self.sequences.append(sequence)
                    self.labels.append(numeric_label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

if __name__ == '__main__':
    dataset = FatigueSequenceDataset('fatigue_data')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = LSTMModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    for epoch in range(epochs):
        for sequences, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'fatigue_lstm.pth')
    print("Model trained and saved to fatigue_lstm.pth")
