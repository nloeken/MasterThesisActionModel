import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Dataset Klasse
class EventDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.features = ['deltaT', 'location_x', 'location_y', 'zone_x', 'zone_y', 'dist_to_goal']
        self.labels = ['act']  # Für Klassifikation
        
        # Aktion Mapping zu Indizes
        self.act2idx = {act: i for i, act in enumerate(self.df['act'].unique())}
        self.idx2act = {i: act for act, i in self.act2idx.items()}
        self.df['act_idx'] = self.df['act'].map(self.act2idx)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.df.iloc[idx][self.features].values, dtype=torch.float)
        y = torch.tensor(self.df.iloc[idx]['act_idx'], dtype=torch.long)
        return x, y

# Einfacher Transformer
class EventTransformer(nn.Module):
    def __init__(self, feature_dim, num_classes, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [seq_len, batch, feature]
        x = self.transformer(x)
        x = self.fc_out(x.squeeze(1))
        return x

# Training Loop
def train_model(train_csv='train.csv', epochs=5, batch_size=32):
    dataset = EventDataset(train_csv)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = EventTransformer(feature_dim=len(dataset.features), num_classes=len(dataset.act2idx))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for X, y in dataloader:
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    train_model()
