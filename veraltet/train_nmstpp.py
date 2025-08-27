from nmstpp import FootballSequenceDataset, NMSTPP360, train_epoch, evaluate, save_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import DataLoader
import torch, torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

# Pfad zu deinem kombinierten CSV anpassen
ds = FootballSequenceDataset('/Users/nloeken/Desktop/open-data/combined/combined_all.csv', seq_len=40, sb360_prefix="sb360_")
le = LabelEncoder()
ds.df["action_cat_enc"] = le.fit_transform(ds.df["action_cat"].astype(str))
ds.df.to_csv("/Users/nloeken/Desktop/open-data/combined/combined_all.csv", index=False)

# einfacher Split in Train/Val
n = len(ds)
train_ds = torch.utils.data.Subset(ds, range(int(n*0.8)))
val_ds = torch.utils.data.Subset(ds, range(int(n*0.8), n))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

model = NMSTPP360(
    action_vocab_size=ds.action_vocab_size,
    action_embed_dim=32,
    num_feat_dim=32,
    sb360_dim_in=len(ds.sb360_cols),
    sb360_proj_dim=64,
    d_model=128,
    nhead=8,
    num_layers=3
).to(device)

model.reset_num_proj(num_input_features=len(ds.features_numeric))
optimizer = optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(1, 6):
    loss = train_epoch(model, train_loader, optimizer, device=device)
    print(f"Epoch {epoch} loss {loss:.4f}")
    metrics = evaluate(model, val_loader, device=device)
    print("Val metrics:", metrics)

save_model(model, "nmstpp360.pt")
