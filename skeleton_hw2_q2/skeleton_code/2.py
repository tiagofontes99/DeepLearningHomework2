import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim

from utils_w_masking import load_rnacompete_data
import utils_w_masking
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set seed for reproducibility
utils_w_masking.configure_seed(42)

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""### LSTM (Long Short-Term Memory) model - Bidiretional"""

from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.layer_dim = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # input_dim deve ser 4 (A,C,G,U) pois os dados são One-Hot
        self.lstm = nn.LSTM(input_dim,
                            hidden_dim//2 if bidirectional else hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True) # If set to True, input/output tensors are provided as (batch, seq_len, features) instead of (seq_len, batch, features)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seq_mask=None):
        # x shape: (Batch, Length, Channels)

        if seq_mask is not None:
            # Compute the real length from the sequence (sum of 1s in the mask)
            lengths = seq_mask.sum(dim=1).cpu().int()

            # Pack the padded sequence (creates a sequence object that ignores the padded elements)
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        output, (hidden, cell) = self.lstm(x)

        if self.bidirectional:
            # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        else:
            hidden = hidden[-1,:,:]     # take the last hidden layer

        hidden = self.dropout(hidden)   # aplica dropout

        return self.fc(hidden)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DotProdAttentionPooling(nn.Module):

    def __init__(self, hidden_dim, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.queries = nn.Parameter(torch.randn(n_heads, hidden_dim))

    def forward(self, context, seq_mask=None):

        B, L, H = context.shape
        Hh = self.n_heads

        context_h = context.unsqueeze(1).expand(B, Hh, L, H).reshape(B * Hh, L, H)

        q = self.queries.unsqueeze(0).expand(B, Hh, H).reshape(B * Hh, 1, H)

        scores = torch.bmm(q, context_h.transpose(1, 2))

        if seq_mask is not None:
            pad_mask = (seq_mask == 0).unsqueeze(1).expand(B, Hh, L).reshape(B * Hh, 1, L)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        alignment = torch.softmax(scores, dim=2)

        pooled = torch.bmm(alignment, context_h).squeeze(1).reshape(B, Hh, H)

        pooled = pooled.reshape(B, Hh * H)

        alignment = alignment.squeeze(1).reshape(B, Hh, L)

        return pooled, alignment


class LSTMWithAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, n_heads=4):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.attn_pool = DotProdAttentionPooling(hidden_dim=hidden_dim, n_heads=n_heads)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * n_heads, output_dim)

    def forward(self, x, seq_mask=None):
        B, L, _ = x.shape

        if seq_mask is not None:
            lengths = seq_mask.sum(dim=1).to(torch.int64).cpu()
            x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.lstm(x_packed)
            out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=L)
        else:
            out, _ = self.lstm(x)

        out = self.dropout(out)

        pooled, alignment = self.attn_pool(out, seq_mask=seq_mask)
        pooled = self.dropout(pooled)
        return self.fc(pooled)




"""### CNN 1D model - o filtro move-se numa única direção (sequência)"""

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(CNN, self).__init__()
        self.num_layers = num_layers

        # define convolution and linear layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=10, padding='same') # sequência de input é (Batch, 4, 41) após permute

        if num_layers > 1:
            self.conv2 = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=5, padding='same')
            conv_out_dim = hidden_dim * 2
        else:
            conv_out_dim = hidden_dim

        # Global Max Pooling (reduz a sequência inteira a um vetor de características)
        self.global_pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(conv_out_dim, output_dim)

    def forward(self, x, seq_mask=None):
        # Input: [Batch, Length, Channels] -> Permute para [Batch, Channels, Length]
        x = x.permute(0, 2, 1)

        x = self.activation(self.conv1(x))
        x = self.dropout(x)

        if self.num_layers > 1:
            x = self.activation(self.conv2(x))
            x = self.dropout(x)

        # Output: [Batch, Channels, Length] -> Global Pool -> [Batch, Channels, 1]
        # o global maxplooling ignora os elementos padding logo não é necessário usar a seq mask
        x = self.global_pool(x).squeeze(-1)  # obter o output por sequência, não por nucleótido
        x = self.dropout(x)
        x = self.fc1(x)

        return x

"""#### Train and Evaluate the models"""

def train(model, train_dataloader, optimizer, criterion):

    epoch_loss = 0

    model.train()

    for batch in tqdm(train_dataloader):
        # Unpack the batch: (sequences, sequence masks, untensities, ValidityMasks)
        x, seqmask, y, mask = batch

        # x shape:          (Batch, 41, 4) - One-Hot Encoded Sequence
        # seq_mask shape:   (Batch, 41)    - 1.0 if valid, 0.0 if NaN
        # y shape:          (Batch, 1)     - Normalized Binding Intensity
        # mask shape:       (Batch, 1)     - 1.0 if valid, 0.0 if NaN

        # Move tensors to gpu
        x = x.to(device)
        seqmask = seqmask.to(device)
        y = y.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # forward pass
        predictions = model(x, seqmask)

        # Calculate Loss - use the mask to zero out invalid data points
        loss = criterion(predictions, y, mask)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_dataloader)

def evaluate(model, eval_dataloader, criterion):

    epoch_loss = 0
    all_preds = []
    all_targets = []
    all_masks = []

    model.eval()

    with torch.no_grad():

        for batch in tqdm(eval_dataloader):
            # Unpack the batch: (Sequences, Intensities, ValidityMasks)
            x, seq_mask, y, mask = batch

            # mover tensores para a gpu
            x = x.to(device)
            seq_mask = seq_mask.to(device)
            y = y.to(device)
            mask = mask.to(device)

            # forward pass
            predictions = model(x, seq_mask)

            # Calculate Loss - use the mask to zero out invalid data points
            loss = criterion(predictions, y, mask)
            epoch_loss += loss.item()

            # guardar tensores na RAM porque a gpu tem memoria limitada (VRAM)
            all_preds.append(predictions.cpu())
            all_targets.append(y.cpu())
            all_masks.append(mask.cpu())

    # concatenar todos os batches num unico tensor grande para usar no spearman
    full_preds = torch.cat(all_preds)
    full_targets = torch.cat(all_targets)
    full_masks = torch.cat(all_masks)

    # calculate the Spearman Correlation
    spearman_corr = utils_w_masking.masked_spearman_correlation(full_preds, full_targets, full_masks)

    return epoch_loss / len(eval_dataloader), spearman_corr

"""#### Load the data"""

# Load Data for a specific protein (e.g., 'RBFOX1', 'PTB', 'A1CF') -> Neste projeto usamos RBFOX1
# This returns a PyTorch TensorDataset ready for training
train_dataset = load_rnacompete_data(protein_name='RBFOX1', split='train')
val_dataset   = load_rnacompete_data(protein_name='RBFOX1', split='val')
test_dataset  = load_rnacompete_data(protein_name='RBFOX1', split='test')

# Wrap in a standard PyTorch DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_epochs = 30

import itertools

hyper_params_LSTM = {
    'num_layers': [1, 2],
    'width': [64, 128],
    'dropout': [0, 0.25],
    'lr_rate': [0.001, 0.0005],
}

# Criar todas as combinações
keys, values = zip(*hyper_params_LSTM.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

best_val_corr_LSTM = -1
best_config_LSTM = None
best_train_losses_LSTM = []
best_valid_losses_LSTM = []

for config in combinations:
    print(f"\n Testing config: {config}")

    model_LSTM = LSTM(
                    input_dim=4,
                    hidden_dim=config['width'],
                    output_dim=1,
                    n_layers=config['num_layers'],
                    bidirectional=True,
                    dropout=config['dropout'],
                )

    # move model to gpu
    model_LSTM = model_LSTM.to(device)

    optimizer = optim.Adam(model_LSTM.parameters(), lr=config['lr_rate'], weight_decay=1e-4)

    criterion = utils_w_masking.masked_mse_loss

    # listas para guardar os resultados para os plots
    train_loss_LSTM = []
    valid_loss_LSTM = []
    valid_corrs_LSTM = []

    # LSTM training loop
    for epoch in range(num_epochs):
        train_loss = train(model_LSTM, train_loader, optimizer, criterion)
        valid_loss, valid_corr = evaluate(model_LSTM, valid_loader, criterion)

        # para os plots
        train_loss_LSTM.append(train_loss)
        valid_loss_LSTM.append(valid_loss)
        valid_corrs_LSTM.append(valid_corr)

        print(f'Epoch: {epoch+1:02}')
        print(f'\t Train Loss: {train_loss:.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f} | Val. Corr: {valid_corr:.4f}')

    # Guardar melhor modelo baseado no Spearman (Validação)
    if valid_corr > best_val_corr_LSTM:
        best_val_corr_LSTM = valid_corr
        best_config_LSTM = config
        best_train_losses_LSTM = train_loss_LSTM[:] # cópia da lista
        best_valid_losses_LSTM = valid_loss_LSTM[:]

print(f"Melhor configuração: {best_config_LSTM} com Spearman: {best_val_corr_LSTM}")

hyper_params_CNN = {
    'num_layers': [1, 2],
    'width': [64, 128],
    'dropout': [0, 0.25],
    'lr_rate': [0.001, 0.0005],
}

# Criar todas as combinações
keys, values = zip(*hyper_params_CNN.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

best_val_corr_CNN = -1
best_config_CNN = None
best_train_losses_CNN = []
best_valid_losses_CNN = []

criterion = utils_w_masking.masked_mse_loss

for config in combinations:
    print(f"\n Testing config: {config}")

    model_CNN = CNN(
                    input_dim=4,
                    hidden_dim=config['width'],
                    output_dim=1,
                    num_layers=config['num_layers'],
                    dropout=config['dropout']
                )

    # move model to gpu
    model_CNN = model_CNN.to(device)

    optimizer = optim.Adam(model_CNN.parameters(), lr=config['lr_rate'], weight_decay=1e-4)

    # listas para guardar os resultados para os plots
    train_loss_CNN = []
    valid_loss_CNN = []
    valid_corrs_CNN = []

    # CNN training loop
    for epoch in range(num_epochs):
        train_loss = train(model_CNN, train_loader, optimizer, criterion)
        valid_loss, valid_corr = evaluate(model_CNN, valid_loader, criterion)

        # para os plots
        train_loss_CNN.append(train_loss)
        valid_loss_CNN.append(valid_loss)
        valid_corrs_CNN.append(valid_corr)

        print(f'Epoch: {epoch+1:02}')
        print(f'\t Train Loss: {train_loss:.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f} | Val. Corr: {valid_corr:.4f}')

    # Guardar melhor modelo baseado no Spearman (Validação)
    if valid_corr > best_val_corr_CNN:
        best_val_corr_CNN = valid_corr
        best_config_CNN = config
        best_train_losses_CNN = train_loss_CNN[:] # cópia da lista
        best_valid_losses_CNN = valid_loss_CNN[:]

print(f"Melhor configuração: {best_config_CNN} com Spearman: {best_val_corr_CNN}")

"""#### Plots for loss on train and validation sets"""

# Plot training and validation losses for the best models
plot_losses_LSTM = {
    'train Loss': best_train_losses_LSTM,
    'val Loss': best_valid_losses_LSTM
}

plot_losses_CNN = {
    'train Loss': best_train_losses_CNN,
    'val Loss': best_valid_losses_CNN
}

# Config string for filenames
config_best_model_LSTM = (
        f"width-{best_config_LSTM['width']}-lr-{best_config_LSTM['lr_rate']}"
        f"-dropout-{best_config_LSTM['dropout']}-layers-{best_config_LSTM['num_layers']}"
    )

config_best_model_CNN = (
        f"width-{best_config_CNN['width']}-lr-{best_config_CNN['lr_rate']}"
        f"-dropout-{best_config_CNN['dropout']}-layers-{best_config_CNN['num_layers']}"
    )

# take the range of epochs for x-axis
epochs_range = range(1, num_epochs + 1)

# Plot and save figures as PDF
utils_w_masking.plot(epochs_range, plot_losses_LSTM, filename=f'loss_LSTM-{config_best_model_LSTM}.pdf', ylim=None)
utils_w_masking.plot(epochs_range, plot_losses_CNN, filename=f'loss_CNN-{config_best_model_CNN}.pdf', ylim=None)

print(best_config_LSTM)
print(best_val_corr_LSTM)
print(best_config_CNN)
print(best_val_corr_CNN)



# LSTM + Attention

criterion = utils_w_masking.masked_mse_loss

cfg = best_config_LSTM
num_epochs_q2 = 30

heads_list = [1, 2, 4, 8]

# LSTM without attention
print("\n----------------------------")
print("LSTM")
print("----------------------------")

model_LSTM_base = LSTM(
    input_dim=4,
    hidden_dim=cfg['width'],
    output_dim=1,
    n_layers=cfg['num_layers'],
    bidirectional=True,
    dropout=cfg['dropout'],
).to(device)

optimizer_base = optim.Adam(
    model_LSTM_base.parameters(),
    lr=cfg['lr_rate'],
    weight_decay=1e-4
)

train_loss_base = []
valid_loss_base = []
valid_corrs_base = []

for epoch in range(num_epochs_q2):
    train_loss = train(model_LSTM_base, train_loader, optimizer_base, criterion)
    valid_loss, valid_corr = evaluate(model_LSTM_base, valid_loader, criterion)

    train_loss_base.append(train_loss)
    valid_loss_base.append(valid_loss)
    valid_corrs_base.append(valid_corr)

    print(f'Epoch: {epoch+1:02}')
    print(f'\t Train Loss: {train_loss:.4f}')
    print(f'\t Val. Loss: {valid_loss:.4f} | Val. Corr: {valid_corr:.4f}')


# LSTM + ATTENTION

print("\n----------------------------")
print("LSTM + ATTENTION")
print("----------------------------")

results_heads = {}

best_val_corr_attn = -1
best_heads = None
best_train_losses_attn = []
best_valid_losses_attn = []

for n_heads in heads_list:
    print(f"\n Attention model with n_heads = {n_heads}")

    model_LSTM_attn = LSTMWithAttention(
        input_dim=4,
        hidden_dim=cfg['width'],
        output_dim=1,
        n_layers=cfg['num_layers'],
        bidirectional=True,
        dropout=cfg['dropout'],
        n_heads=n_heads
    ).to(device)

    optimizer_attn = optim.Adam(
        model_LSTM_attn.parameters(),
        lr=cfg['lr_rate'],
        weight_decay=1e-4
    )

    train_loss_attn = []
    valid_loss_attn = []
    valid_corrs_attn = []

    for epoch in range(num_epochs_q2):
        train_loss = train(model_LSTM_attn, train_loader, optimizer_attn, criterion)
        valid_loss, valid_corr = evaluate(model_LSTM_attn, valid_loader, criterion)

        train_loss_attn.append(train_loss)
        valid_loss_attn.append(valid_loss)
        valid_corrs_attn.append(valid_corr)

        print(f'Epoch: {epoch+1:02}')
        print(f'\t Train Loss: {train_loss:.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f} | Val. Corr: {valid_corr:.4f}')

    results_heads[n_heads] = {
        'train_loss': train_loss_attn,
        'valid_loss': valid_loss_attn,
        'valid_corr': valid_corrs_attn
    }

    # Choses the best nº heads
    if valid_corr > best_val_corr_attn:
        best_val_corr_attn = valid_corr
        best_heads = n_heads
        best_train_losses_attn = train_loss_attn[:]
        best_valid_losses_attn = valid_loss_attn[:]

print("\n----------------------------")
print(f"Best Attention Model: n_heads = {best_heads} | Val. Spearman = {best_val_corr_attn:.4f}")
print("----------------------------")




# Plots of the model with each number of heads

epochs = list(range(1, num_epochs_q2 + 1))

# LSTM loss curves without attention
train_loss_LSTM_base = train_loss_base
valid_loss_LSTM_base = valid_loss_base

for n_heads, res in results_heads.items():

    train_loss_LSTM_attn = res["train_loss"]
    valid_loss_LSTM_attn = res["valid_loss"]

    plt.figure()
    plt.plot(epochs, train_loss_LSTM_base, label="train (baseline)")
    plt.plot(epochs, valid_loss_LSTM_base, label="val (baseline)")
    plt.plot(epochs, train_loss_LSTM_attn, label=f"train (attention, heads={n_heads})")
    plt.plot(epochs, valid_loss_LSTM_attn, label=f"val (attention, heads={n_heads})")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"LSTM baseline vs LSTM + attention (heads={n_heads})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"Q2_LSTM_attention_loss_heads_{n_heads}.pdf")
    plt.show()



# Performance (Baseline vs Attention heads=8)

import torch
import torch.optim as optim

criterion = utils_w_masking.masked_mse_loss

cfg = best_config_LSTM
num_epochs_q2 = 30

chosen_heads = 8

# LSTM without attention

print("\n----------------------------")
print("LSTM (final run)")
print("----------------------------")

model_LSTM_base = LSTM(
    input_dim=4,
    hidden_dim=cfg['width'],
    output_dim=1,
    n_layers=cfg['num_layers'],
    bidirectional=True,
    dropout=cfg['dropout'],
).to(device)

optimizer_base = optim.Adam(
    model_LSTM_base.parameters(),
    lr=cfg['lr_rate'],
    weight_decay=1e-4
)

train_loss_base = []
valid_loss_base = []
valid_corrs_base = []

for epoch in range(num_epochs_q2):
    train_loss = train(model_LSTM_base, train_loader, optimizer_base, criterion)
    valid_loss, valid_corr = evaluate(model_LSTM_base, valid_loader, criterion)

    train_loss_base.append(train_loss)
    valid_loss_base.append(valid_loss)
    valid_corrs_base.append(valid_corr)

    print(f'Epoch: {epoch+1:02}')
    print(f'\t Train Loss: {train_loss:.4f}')
    print(f'\t Val. Loss: {valid_loss:.4f} | Val. Corr: {valid_corr:.4f}')


# LSTM + ATTENTION (heads=8)

print("\n----------------------------")
print(f"LSTM + ATTENTION (heads={chosen_heads}) (final run)")
print("----------------------------")

model_LSTM_attn = LSTMWithAttention(
    input_dim=4,
    hidden_dim=cfg['width'],
    output_dim=1,
    n_layers=cfg['num_layers'],
    bidirectional=True,
    dropout=cfg['dropout'],
    n_heads=chosen_heads
).to(device)

optimizer_attn = optim.Adam(
    model_LSTM_attn.parameters(),
    lr=cfg['lr_rate'],
    weight_decay=1e-4
)

train_loss_attn = []
valid_loss_attn = []
valid_corrs_attn = []

for epoch in range(num_epochs_q2):
    train_loss = train(model_LSTM_attn, train_loader, optimizer_attn, criterion)
    valid_loss, valid_corr = evaluate(model_LSTM_attn, valid_loader, criterion)

    train_loss_attn.append(train_loss)
    valid_loss_attn.append(valid_loss)
    valid_corrs_attn.append(valid_corr)

    print(f'Epoch: {epoch+1:02}')
    print(f'\t Train Loss: {train_loss:.4f}')
    print(f'\t Val. Loss: {valid_loss:.4f} | Val. Corr: {valid_corr:.4f}')


# TEST

print("\n----------------------------")
print("Test")
print("----------------------------")

test_loss_base, test_corr_base = evaluate(model_LSTM_base, test_loader, criterion)
test_loss_attn, test_corr_attn = evaluate(model_LSTM_attn, test_loader, criterion)

print(f"LSTM            | Test Spearman: {test_corr_base:.4f}")
print(f"LSTM + Attention (heads={chosen_heads}) | Test Spearman: {test_corr_attn:.4f}")
