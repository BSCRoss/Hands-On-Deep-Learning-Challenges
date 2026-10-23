import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from sklearn.metrics import f1_score
from copy import deepcopy
import math
from torch.utils.data import Dataset
from typing import Any

device = "cuda" if torch.cuda.is_available() else "cpu"


class MazeConv(MessagePassing):

    def __init__(self, hidden_dim: int):
        super().__init__(aggr='add')
        in_msg = hidden_dim * 3   # x_i, x_j, x_j - x_i

        self.msg_mlp = nn.Sequential(
            nn.Linear(in_msg, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        # GRU cell for update: treats aggregated message as "input", x as "hidden"
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        agg = self.propagate(edge_index, x=x)
        out = self.gru(agg, x)          # gated update
        return self.norm(out)           # no explicit residual; GRU handles it

    def message(self, x_i, x_j):
        return self.msg_mlp(torch.cat([x_i, x_j, x_j - x_i], dim=-1))



class MazeGNN(nn.Module):
    def __init__(self, hidden_dim: int = 96):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoder: raw node features (2-dim) → hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.conv = MazeConv(hidden_dim)

        self.skip_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Decoder: hidden_dim → 2 class logits
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data, num_nodes: int):
        x, edge_index = data.x.float(), data.edge_index

        # Encode raw features; keep as permanent "endpoint memory"
        inp = self.encoder(x)       # [N, H]
        h   = inp.clone()


        n_rounds = max(8, int(2.5 * math.sqrt(num_nodes)) + 4)

        for _ in range(n_rounds):

            h = self.skip_proj(torch.cat([inp, h], dim=-1))
            h = self.conv(h, edge_index)

        return F.log_softmax(self.decoder(h), dim=1)



def init_model() -> nn.Module:
    model = MazeGNN(hidden_dim=96)
    return model


def train_model(model: nn.Module, train_dataset: Dataset) -> nn.Module:

    small = [g for g in train_dataset if g.num_nodes == 16]
    if len(small) == 0:
        small = list(train_dataset)

    # Use more training samples if available — augment by using full set
    val_size  = max(1, int(0.15 * len(small)))
    train_set = small[:-val_size]
    val_set   = small[-val_size:]

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=1)

    criterion = nn.NLLLoss()

    # Two-phase training: warm up with higher LR, then fine-tune
    epochs_phase1 = 60
    epochs_phase2 = 40
    epochs = epochs_phase1 + epochs_phase2

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=5e-6
    )

    model.to(device)
    best_model  = deepcopy(model)
    best_metric = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data, data.num_nodes)
            loss = criterion(pred, data.y.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # ---- Validation ----
        model.eval()
        correct, total, perfect, graphs = 0, 0, 0, 0
        all_pred, all_true = [], []

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data, data.num_nodes)
                yhat = pred.argmax(dim=1)
                n = data.num_nodes
                correct += (yhat == data.y).sum().item()
                total   += n
                graphs  += 1
                if (yhat == data.y).all():
                    perfect += 1
                all_pred.extend(yhat.cpu().tolist())
                all_true.extend(data.y.cpu().tolist())

        f1     = f1_score(all_true, all_pred, zero_division=0)
        g_acc  = perfect / graphs
        metric = 0.5 * f1 + 0.5 * g_acc   # balanced metric

        if metric > best_metric:
            best_metric = metric
            best_model  = deepcopy(model)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d} | loss {running_loss/len(train_loader):.4f} "
                f"| f1 {f1:.3f} | node_acc {correct/total:.3f} | graph_acc {g_acc:.3f}"
            )

    model.cpu()
    best_model.cpu()
    return best_model