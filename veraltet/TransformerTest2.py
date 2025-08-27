import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ======================
# 1) Dataset
# ======================
class EventDataset(Dataset):
    def __init__(self, df, feature_cols, label_col="next_action_cat", seq_len=20):
        """
        df: DataFrame mit Events (inkl. Features und Labels)
        feature_cols: Liste der Spalten, die als Input benutzt werden
        label_col: Spalte für Zielvariable (nächste Aktion)
        seq_len: Länge der Sequenzen (wird ggf. gepaddet)
        """
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.seq_len = seq_len

        # Features + Labels vorbereiten
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.labels = self.df[label_col].values.astype(np.int64)

        # Sliding Windows für Sequenzen
        self.indices = [i for i in range(len(self.df) - seq_len - 1)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len

        # Input-Sequenz (Feature-Vektoren)
        seq_x = self.features[start_idx:end_idx]

        # Label: nächste Aktion nach der Sequenz
        y = self.labels[end_idx]

        # Maske (alle gültigen Tokens = 1, Padding = 0)
        mask = np.ones(self.seq_len, dtype=np.bool_)

        return (
            torch.tensor(seq_x, dtype=torch.float32),  # [seq_len, input_dim]
            torch.tensor(mask, dtype=torch.bool),      # [seq_len]
            torch.tensor(y, dtype=torch.long)          # int Label
        )


# ======================
# 2) Modell
# ======================
class ActionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_actions, max_len=512):
        super(ActionTransformer, self).__init__()
        
        # Feature -> Embedding
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)

        # Positional Encoding
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Klassifikationskopf
        self.fc_out = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, input_dim]
        mask: [batch_size, seq_len]  (1=keep, 0=pad)
        """
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        x = self.feature_embedding(x) + self.pos_embedding(positions)
        x = self.transformer_encoder(x, src_key_padding_mask=~mask if mask is not None else None)

        # Letztes Event -> Vorhersage der nächsten Aktion
        last_hidden = x[:, -1, :]
        out = self.fc_out(last_hidden)

        return out


# ======================
# 3) Training
# ======================
def train_model(model, dataloader, num_epochs=10, lr=1e-4, device="cuda"):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for features, mask, labels in dataloader:
            features, mask, labels = features.to(device), mask.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(features, mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")


# ======================
# 4) Beispiel Nutzung
# ======================
if __name__ == "__main__":
    import pandas as pd

    # Beispiel: Dummy-Daten (normalerweise: dein df nach Feature Engineering)
    n_samples = 1000
    df = pd.DataFrame({
        "x": np.random.rand(n_samples),
        "y": np.random.rand(n_samples),
        "distance_to_goal": np.random.rand(n_samples),
        "time_since_last_event": np.random.rand(n_samples),
        "player_under_pressure": np.random.randint(0, 2, size=n_samples),
        "prev_action_cat": np.random.randint(0, 5, size=n_samples),
        "team_name": np.random.randint(0, 3, size=n_samples),
        "position_name": np.random.randint(0, 4, size=n_samples),
        "next_action_cat": np.random.randint(0, 6, size=n_samples)  # 6 mögliche Aktionen
    })

    feature_cols = [
        "x", "y", "distance_to_goal",
        "time_since_last_event", "player_under_pressure", "prev_action_cat",
        "team_name", "position_name"
    ]

    dataset = EventDataset(df, feature_cols=feature_cols, label_col="next_action_cat", seq_len=20)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Modell
    input_dim = len(feature_cols)
    hidden_dim = 64
    num_heads = 4
    num_layers = 2
    num_actions = df["next_action_cat"].nunique()

    model = ActionTransformer(input_dim, hidden_dim, num_heads, num_layers, num_actions)

    # Training
    train_model(model, dataloader, num_epochs=5, lr=1e-3, device="cpu")
