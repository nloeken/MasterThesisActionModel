"""
nmstpp_model.py

Transformer-based Neural Marked Spatio-Temporal Point Process (NMSTPP) - simplified,
adapted for: next action prediction + success prediction using event + 360 data.

How to use (high level):
- Prepare your combined CSV (the same as your pipeline produces) with the following
  columns (or adapt the names inside the Dataset class):
    - match_id, event_index (monotonic per match), action_cat_enc (int), event_success (0/1),
      x, y, time_seconds, <any other numeric contextual features>,
      optionally: sb360_X features (flattened freeze_frame / 360 features)
    - target columns: next_action_cat_enc, next_action_success (the user already creates those)
- Instantiate FootballSequenceDataset(combo_csv, seq_len=40, ...)
- Create train/val/test DataLoaders
- Instantiate NMSTPP360(...)
- Call train_epoch / evaluate functions provided below
"""

import os
from typing import List, Optional, Tuple, Dict
import json
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------- Dataset -------------------------------------

class FootballSequenceDataset(Dataset):
    """
    Build sequences of fixed length (seq_len) per possession or per match.
    Expects a combined CSV that already contains engineered features and integer-encoded
    action categories (action_cat_enc) and binary event_success.
    IMPORTANT: Adapt column names below if your CSV uses different ones.
    """
    def __init__(self,
                 combined_csv: str,
                 seq_len: int = 40,
                 stride: int = 1,
                 features_numeric: Optional[List[str]] = None,
                 sb360_prefix: Optional[str] = None,
                 device: str = 'cpu'):
        """
        :param combined_csv: path to your combined CSV (the file your pipeline saves)
        :param seq_len: number of past events to feed the model
        :param stride: sliding window stride (1 for all subsequences)
        :param features_numeric: list of numeric column names to include as inputs (besides action_cat_enc)
        :param sb360_prefix: if your 360 features are flattened as sb360_0, sb360_1, ... set this prefix
        """
        self.df = pd.read_csv(combined_csv)
        self.seq_len = seq_len
        self.stride = stride
        self.device = device

        # Default numeric features if not provided
        if features_numeric is None:
            # these match the user's feature engineering in their XGBoost pipeline
            features_numeric = ['x', 'y', 'distance_to_goal', 'angle_to_goal', 'in_box',
                                'in_cross_zone', 'nearby_opponents', 'high_pressure', 'low_pressure',
                                'orientation', 'free_teammates', 'time_seconds', 'is_late_game',
                                'is_losing', 'duration', 'possession_change', 'combo_depth',
                                'time_since_last_event', 'progress_to_goal']
        self.features_numeric = [c for c in features_numeric if c in self.df.columns]

        # detect flattened 360 features automatically if prefix provided or detect pattern
        self.sb360_cols = []
        if sb360_prefix:
            self.sb360_cols = [c for c in self.df.columns if c.startswith(sb360_prefix)]
        else:
            # heuristics: look for 'sb360' or '360' or 'freeze_frame_flat' style columns
            possible = [c for c in self.df.columns if ('sb360' in c) or (c.startswith('three_sixty')) or ('360' in c and c.count('_')>0)]
            self.sb360_cols = possible

        # required columns
        required = ['match_id', 'action_cat_enc', 'event_success', 'next_action_cat_enc', 'next_action_success']
        for r in required:
            if r not in self.df.columns:
                raise ValueError(f"Required column '{r}' not found in combined csv. Present columns: {list(self.df.columns)[:40]}")

        # We'll build sequences per match_id to avoid crossing match boundaries
        self.sequences = []
        grouped = self.df.sort_values(['match_id']).groupby('match_id')
        for match_id, group in tqdm(grouped, desc='Building sequences'):
            group = group.reset_index(drop=True)
            N = len(group)
            for start in range(0, max(0, N - 1), self.stride):
                end = start + self.seq_len
                if end >= N - 1:
                    # ensure we have a valid next event target at index end (next event)
                    if start + 1 >= N:
                        break
                    # we take the last seq_len events if available
                    s = max(0, N - self.seq_len)
                    subseq = group.iloc[s:N]
                    next_idx = N - 1
                else:
                    subseq = group.iloc[start:end]
                    next_idx = end  # next event to predict
                # build inputs and targets
                x_actions = subseq['action_cat_enc'].fillna(0).astype(int).values
                x_num = subseq[self.features_numeric].fillna(0).astype(float).values if len(self.features_numeric)>0 else np.zeros((len(subseq),0))
                x_sb360 = subseq[self.sb360_cols].fillna(0).astype(float).values if len(self.sb360_cols)>0 else np.zeros((len(subseq),0))

                target_action = int(group.loc[next_idx, 'next_action_cat_enc']) if not pd.isna(group.loc[next_idx, 'next_action_cat_enc']) else -1
                target_success = int(group.loc[next_idx, 'next_action_success']) if not pd.isna(group.loc[next_idx, 'next_action_success']) else -1

                # filter invalid targets
                if target_action < 0 or target_success < 0:
                    continue

                self.sequences.append({
                    'match_id': match_id,
                    'x_actions': x_actions.astype(np.int64),
                    'x_num': x_num.astype(np.float32),
                    'x_sb360': x_sb360.astype(np.float32),
                    'target_action': target_action,
                    'target_success': target_success
                })

        # compute vocab size for action categories
        self.action_vocab_size = int(self.df['action_cat_enc'].max() + 1)
        if self.action_vocab_size <= 0:
            raise ValueError("action_cat_enc appears invalid (max <= 0)")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        # pad sequences to seq_len
        L = len(s['x_actions'])
        if L < self.seq_len:
            pad_len = self.seq_len - L
            actions = np.concatenate([np.zeros(pad_len, dtype=np.int64), s['x_actions']], axis=0)
            num = np.vstack([np.zeros((pad_len, s['x_num'].shape[1]), dtype=np.float32), s['x_num']]) if s['x_num'].shape[1]>0 else np.zeros((self.seq_len,0),dtype=np.float32)
            sb360 = np.vstack([np.zeros((pad_len, s['x_sb360'].shape[1]), dtype=np.float32), s['x_sb360']]) if s['x_sb360'].shape[1]>0 else np.zeros((self.seq_len,0),dtype=np.float32)
        else:
            actions = s['x_actions'][-self.seq_len:]
            num = s['x_num'][-self.seq_len:]
            sb360 = s['x_sb360'][-self.seq_len:]
        return {
            'actions': torch.tensor(actions, dtype=torch.long),
            'num': torch.tensor(num, dtype=torch.float32),
            'sb360': torch.tensor(sb360, dtype=torch.float32),
            'target_action': torch.tensor(s['target_action'], dtype=torch.long),
            'target_success': torch.tensor(s['target_success'], dtype=torch.float32)
        }

# ----------------------------- Model ---------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class NMSTPP360(nn.Module):
    """
    Simplified NMSTPP-like model:
    - action token embedding for discrete past actions
    - optional numeric contextual features concatenated per timestep
    - optional 360 features concatenated per timestep (after linear reduction)
    - Transformer encoder over sequence
    - pooled representation -> prediction heads:
        - next action (categorical, cross-entropy)
        - next success (binary, BCEWithLogits)
    """
    def __init__(self,
                 action_vocab_size: int,
                 action_embed_dim: int = 32,
                 num_feat_dim: int = 16,
                 sb360_dim_in: int = 0,
                 sb360_proj_dim: int = 32,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 num_actions_out: Optional[int] = None):
        super().__init__()
        self.action_vocab_size = action_vocab_size
        self.action_embed_dim = action_embed_dim

        # embeddings
        self.action_embedding = nn.Embedding(action_vocab_size, action_embed_dim, padding_idx=0)

        # numeric features projection per timestep
        self.num_feat_dim = num_feat_dim
        self.num_proj = nn.Linear(0, num_feat_dim) if num_feat_dim>0 else None  # will be re-created after we know input size

        # sb360 projection
        self.sb360_dim_in = sb360_dim_in
        if sb360_dim_in>0:
            self.sb360_proj = nn.Sequential(
                nn.Linear(sb360_dim_in, sb360_proj_dim),
                nn.ReLU(),
                nn.LayerNorm(sb360_proj_dim)
            )
        else:
            self.sb360_proj = None

        # concatenated timestep embedding size
        timestep_dim = action_embed_dim + (num_feat_dim if num_feat_dim>0 else 0) + (sb360_proj_dim if sb360_dim_in>0 else 0)
        self.input_proj = nn.Linear(timestep_dim, d_model) if timestep_dim != d_model else nn.Identity()

        # positional encoding
        self.pos_enc = PositionalEncoding(d_model)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # pooling: use mean pooling over sequence dimension
        self.pool = lambda x: x.mean(dim=1)

        # output heads
        self.num_actions_out = num_actions_out if num_actions_out is not None else action_vocab_size
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.LayerNorm(d_model//2),
            nn.Linear(d_model//2, self.num_actions_out)
        )
        self.success_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.LayerNorm(d_model//2),
            nn.Linear(d_model//2, 1)
        )

    def reset_num_proj(self, num_input_features):
        "Call this when you know the numeric feature vector size per timestep."
        if num_input_features > 0:
            self.num_proj = nn.Linear(num_input_features, self.num_feat_dim)
        else:
            self.num_proj = None

    def forward(self, actions: torch.LongTensor, num_feats: torch.FloatTensor, sb360: torch.FloatTensor):
        """
        :param actions: (batch, seq_len) int tokens
        :param num_feats: (batch, seq_len, num_input_features)
        :param sb360: (batch, seq_len, sb360_dim_in)
        :return: action_logits (batch, num_actions_out), success_logits (batch,)
        """
        B, S = actions.size()
        # action embedding
        a_emb = self.action_embedding(actions)  # (B, S, action_embed_dim)

        # numeric features
        if self.num_proj is not None and num_feats is not None and num_feats.size(-1)>0:
            n = self.num_proj(num_feats)  # (B, S, num_feat_dim)
        else:
            n = torch.zeros((B, S, 0), device=actions.device)

        # sb360 projection
        if self.sb360_proj is not None and sb360 is not None and sb360.size(-1)>0:
            s = self.sb360_proj(sb360)  # (B, S, sb360_proj_dim)
        else:
            s = torch.zeros((B, S, 0), device=actions.device)

        # concat per timestep
        emb = torch.cat([a_emb, n, s], dim=-1)  # (B, S, emb_dim)
        if isinstance(self.input_proj, nn.Identity):
            x = emb
        else:
            x = self.input_proj(emb)  # (B, S, d_model)

        # add positional encoding and transformer
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, S, d_model)

        # pool over sequence
        pooled = self.pool(x)  # (B, d_model)

        # heads
        action_logits = self.action_head(pooled)  # (B, num_actions_out)
        success_logits = self.success_head(pooled).squeeze(-1)  # (B,)

        return action_logits, success_logits

# ----------------------------- Training & Eval --------------------------------

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer, device='cpu', clf_weight: float = 1.0):
    model.train()
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        actions = batch['actions'].to(device)
        num = batch['num'].to(device)
        sb360 = batch['sb360'].to(device)
        target_action = batch['target_action'].to(device)
        target_success = batch['target_success'].to(device).float()

        optimizer.zero_grad()
        logits_action, logits_success = model(actions, num, sb360)
        loss_action = ce(logits_action, target_action)
        loss_success = bce(logits_success, target_success)
        loss = loss_action + 0.5 * loss_success  # weight success less (tunable)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * actions.size(0)
    return total_loss / len(dataloader.dataset)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate(model: nn.Module, dataloader: DataLoader, device='cpu') -> Dict[str, float]:
    model.eval()
    preds_action = []
    trues_action = []
    preds_success_prob = []
    trues_success = []
    with torch.no_grad():
        for batch in dataloader:
            actions = batch['actions'].to(device)
            num = batch['num'].to(device)
            sb360 = batch['sb360'].to(device)
            target_action = batch['target_action'].to(device)
            target_success = batch['target_success'].to(device).float()

            logits_action, logits_success = model(actions, num, sb360)
            probs_action = torch.softmax(logits_action, dim=-1)
            pred_action = torch.argmax(probs_action, dim=-1).cpu().numpy()
            pred_success_prob = torch.sigmoid(logits_success).cpu().numpy()

            preds_action.extend(pred_action.tolist())
            trues_action.extend(target_action.cpu().numpy().tolist())
            preds_success_prob.extend(pred_success_prob.tolist())
            trues_success.extend(target_success.cpu().numpy().tolist())

    # metrics
    action_acc = accuracy_score(trues_action, preds_action)
    action_f1 = f1_score(trues_action, preds_action, average='weighted')
    try:
        success_auc = roc_auc_score(trues_success, preds_success_prob)
    except Exception:
        success_auc = float('nan')
    success_pred_binary = [1 if p>=0.5 else 0 for p in preds_success_prob]
    success_acc = accuracy_score(trues_success, success_pred_binary)
    success_f1 = f1_score(trues_success, success_pred_binary, average='binary')

    return {
        'action_acc': action_acc,
        'action_f1': action_f1,
        'success_auc': success_auc,
        'success_acc': success_acc,
        'success_f1': success_f1
    }

# ----------------------------- Save / Load ---------------------------------

def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)

def load_model(model: nn.Module, path: str, map_location='cpu'):
    model.load_state_dict(torch.load(path, map_location=map_location))
    model.eval()

# ----------------------------- Example usage --------------------------------
if __name__ == "__main__":
    print("This module defines NMSTPP360 model and FootballSequenceDataset. Import from your training script.")
