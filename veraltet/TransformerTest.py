import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FootballDataset(Dataset):
    def __init__(self, df, le_action, max_len=50):
        self.max_len = max_len
        self.le_action = le_action
        
        # gruppiere nach Ballbesitz
        self.sequences = []
        for _, poss in df.groupby("possession"):
            # baue Feature-Arrays
            actions = poss["action_cat_enc"].values
            positions = poss["position_name_enc"].values
            teams = poss["team_name_enc"].values
            num_feats = poss[[
                'x', 'y', 'distance_to_goal', 'angle_to_goal',
                'in_box', 'in_flank_zone', 'nearby_opponents',
                'high_pressure', 'low_pressure', 'free_teammates'
            ]].values.astype(np.float32)
            next_action = poss["next_action_cat_enc"].values
            next_success = poss["next_action_success"].values.astype(np.float32)

            # skip zu kurze Sequenzen
            if len(actions) < 2:
                continue

            self.sequences.append({
                "actions": actions,
                "positions": positions,
                "teams": teams,
                "num_feats": num_feats,
                "target_action": next_action,
                "target_success": next_success
            })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        L = len(seq["actions"])
        
        # Padding
        pad_len = self.max_len - L if L < self.max_len else 0
        def pad(arr, pad_value=0):
            if pad_len > 0:
                return np.pad(arr, ((0,pad_len),(0,0)), constant_values=pad_value) if arr.ndim==2 else np.pad(arr, (0,pad_len), constant_values=pad_value)
            else:
                return arr[-self.max_len:]
        
        actions = pad(seq["actions"])
        positions = pad(seq["positions"])
        teams = pad(seq["teams"])
        num_feats = pad(seq["num_feats"])
        target_action = pad(seq["target_action"])
        target_success = pad(seq["target_success"])

        # Maske: 1 = echtes Event, 0 = Padding
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:min(L, self.max_len)] = 1

        return {
            "actions": torch.tensor(actions, dtype=torch.long),
            "positions": torch.tensor(positions, dtype=torch.long),
            "teams": torch.tensor(teams, dtype=torch.long),
            "num_feats": torch.tensor(num_feats, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "target_action": torch.tensor(target_action, dtype=torch.long),
            "target_success": torch.tensor(target_success, dtype=torch.float32),
        }
    
import torch.nn as nn

class FootballTransformer(nn.Module):
    def __init__(self, num_actions, num_positions, num_teams, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, d_model)
        self.position_emb = nn.Embedding(num_positions, d_model)
        self.team_emb = nn.Embedding(num_teams, d_model)
        self.num_feat_proj = nn.Linear(10, d_model)  # numerische Features -> embedding

        self.pos_encoding = nn.Parameter(torch.randn(1, 200, d_model))  # max_len=200

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_action = nn.Linear(d_model, num_actions)
        self.fc_success = nn.Linear(d_model, 1)

    def forward(self, actions, positions, teams, num_feats, mask):
        x = (
            self.action_emb(actions) +
            self.position_emb(positions) +
            self.team_emb(teams) +
            self.num_feat_proj(num_feats)
        )
        x = x + self.pos_encoding[:, :x.size(1), :]
        # src_key_padding_mask erwartet: True=Padding
        pad_mask = mask == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        
        # Event-weise Vorhersagen
        action_logits = self.fc_action(x)
        success_logits = self.fc_success(x).squeeze(-1)
        return action_logits, success_logits

import torch.optim as optim
import torch.nn.functional as F

def train_transformer(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cuda"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            actions = batch["actions"].to(device)
            positions = batch["positions"].to(device)
            teams = batch["teams"].to(device)
            num_feats = batch["num_feats"].to(device)
            mask = batch["mask"].to(device)
            target_action = batch["target_action"].to(device)
            target_success = batch["target_success"].to(device)

            optimizer.zero_grad()
            pred_actions, pred_success = model(actions, positions, teams, num_feats, mask)

            # Maskierte Loss (nur echte Events, kein Padding)
            active = mask.view(-1) == 1
            action_loss = F.cross_entropy(
                pred_actions.view(-1, pred_actions.size(-1))[active],
                target_action.view(-1)[active]
            )
            success_loss = F.binary_cross_entropy_with_logits(
                pred_success.view(-1)[active],
                target_success.view(-1)[active]
            )
            loss = action_loss + success_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f}")

        # optional: simple validation
        evaluate_transformer(model, val_loader, device)


@torch.no_grad()
def evaluate_transformer(model, loader, device="cuda"):
    model.eval()
    total_correct, total, total_success_correct = 0, 0, 0
    all_preds, all_labels = [], []
    for batch in loader:
        actions = batch["actions"].to(device)
        positions = batch["positions"].to(device)
        teams = batch["teams"].to(device)
        num_feats = batch["num_feats"].to(device)
        mask = batch["mask"].to(device)
        target_action = batch["target_action"].to(device)
        target_success = batch["target_success"].to(device)

        pred_actions, pred_success = model(actions, positions, teams, num_feats, mask)
        pred_actions = pred_actions.argmax(dim=-1)

        active = mask.view(-1) == 1
        total_correct += (pred_actions.view(-1)[active] == target_action.view(-1)[active]).sum().item()
        total += active.sum().item()

    print(f"Validation Action Accuracy: {total_correct/total:.3f}")

# Dataset
train_ds = FootballDataset(train_df, le_action, max_len=50)
val_ds = FootballDataset(val_df, le_action, max_len=50)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Modell
model = FootballTransformer(
    num_actions=len(le_action.classes_),
    num_positions=df['position_name_enc'].nunique(),
    num_teams=df['team_name_enc'].nunique()
)

# Training
train_transformer(model, train_loader, val_loader, epochs=20)

