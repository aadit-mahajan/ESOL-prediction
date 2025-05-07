from model import twoTrackNetwork
from data import ESOLDataset
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from torch.utils.data import random_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import os 

torch.manual_seed(42)

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for graph_data, cls_embed, mean_embed, labels in train_loader:
        graph_data = graph_data.to(device)
        cls_embed = cls_embed.to(device)
        mean_embed = mean_embed.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(graph_data, cls_embed, mean_embed)
        loss = criterion(outputs.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for graph_data, cls_embed, mean_embed, labels in val_loader:
            graph_data = graph_data.to(device)
            cls_embed = cls_embed.to(device)
            mean_embed = mean_embed.to(device)
            labels = labels.to(device)

            outputs = model(graph_data, cls_embed, mean_embed)
            loss = criterion(outputs.view(-1), labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(val_loader)

def evaluate_regression(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()

    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaNs detected in y_true or y_pred!")
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValueError("Infs detected in y_true or y_pred!")
    assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Pearson r: {pearson_r:.4f}")
    print(f"Spearman ρ: {spearman_r:.4f}")

    return f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}\nPearson r: {pearson_r:.4f}\nSpearman ρ: {spearman_r:.4f}"

def make_plots(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', lw=2)  # Line y=x
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs True Values")
    plt.savefig("plots/predictions.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    residuals = y_true.flatten() - y_pred.flatten()
    plt.scatter(y_pred.flatten(), residuals)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.savefig("plots/residuals.png")
    plt.close()


def main():
    plots_path = "./plots"
    os.makedirs(plots_path, exist_ok=True)
    dataset_path = "./esol.parquet"
    df = pd.read_parquet(dataset_path)
    print("Dataset loaded\n")

    df['X_cls'] = df['X_cls'].apply(lambda x: np.array(x))
    df['X_mean'] = df['X_mean'].apply(lambda x: np.array(x))

    # Create dataset
    dataset = ESOLDataset(df)
    # split the dataset
    train, test, val = random_split(
        dataset, 
        lengths=[0.75, 0.1, 0.15],
        generator=torch.Generator().manual_seed(42)
    )

    trainLoader = DataLoader(train, batch_size=32, shuffle=True)
    testLoader = DataLoader(test, batch_size=32, shuffle=False)
    valLoader = DataLoader(val, batch_size=32, shuffle=False)

    in_channels_cls = df['X_cls'].iloc[0].shape[0]
    in_channels_mean = df['X_mean'].iloc[0].shape[0]
    in_channels = dataset[0][0].x.shape[1]
    out_channels = 16

    print(f"Input channels: {in_channels_cls}, {in_channels_mean}, {in_channels}")
    print(f"Output channels: {out_channels}")

    print("Loaders created. Starting training...\n")
    model = twoTrackNetwork(
        in_channels=in_channels,
        in_channels_cls=in_channels_cls, 
        in_channels_mean=in_channels_mean,
        out_channels=out_channels
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    criterion = torch.nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epochs = 20
    history = []
    for epoch in range(epochs):
        train_loss = train_model(model, trainLoader, optimizer, criterion, device)
        val_loss = evaluate_model(model, valLoader, criterion, device)

        losses = {
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        print(f'epoch: {epoch}\ntrain loss: {losses['train_loss']}, val loss: {losses['val_loss']}\n')
        history.append(losses)

    print("Training Complete")

    losses_df = pd.DataFrame(history)
    plt.figure(figsize=(8, 6))
    plt.title("Losses")
    sns.lineplot(losses_df)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.xticks(range(0, epochs, 1))
    plt.grid()
    plt.savefig("plots/losses.png") 

    # test the model on the test data:
    test_loss = evaluate_model(model, testLoader, criterion, device)
    print(f'Test loss: {test_loss}')
    y_true = []
    y_pred = []
    with torch.no_grad():
        for graph_data, cls_embed, mean_embed, labels in testLoader:
            graph_data = graph_data.to(device)
            cls_embed = cls_embed.to(device)
            mean_embed = mean_embed.to(device)
            labels = labels.to(device)

            outputs = model(graph_data, cls_embed, mean_embed)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("Evaluation metrics:")
    output = evaluate_regression(y_true, y_pred)
    with open("perf_mets.txt", 'w+') as f:
        f.write(output)

    make_plots(y_true, y_pred)
    print("Plots saved as predictions.png and residuals.png")

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

if __name__ == '__main__':
    main()





    


