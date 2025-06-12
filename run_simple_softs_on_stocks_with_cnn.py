import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from exp.exp_custom import Exp_Custom
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_basic import Exp_Basic

# fix seed for reproducibility
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.set_num_threads(6)

# CNN Model Classes (from model.py)
class TimeSeries(Dataset):
    def __init__(self, data, labels, window_size=30):
        self.data = data
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]
        y = self.labels[idx+self.window_size]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y, dtype=torch.long)

class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(84, 250)
        self.fc2 = nn.Linear(250, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# Enhanced CNN for multi-step prediction
class EnhancedCNN1D(nn.Module):
    def __init__(self, input_dim=32, seq_len=30, pred_len=14):
        super(EnhancedCNN1D, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers for multi-step prediction
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, input_dim * pred_len)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global average pooling
        x = self.pool(x)  # (batch_size, 64, 1)
        x = x.squeeze(-1)  # (batch_size, 64)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Reshape to (batch_size, pred_len, input_dim)
        x = x.view(-1, self.pred_len, self.input_dim)
        
        return x

class MultiStepDataset(Dataset):
    def __init__(self, data, seq_len=30, pred_len=14):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# basic config
config = {
    # dataset settings
    'root_path': 'data/processed/',
    'data_path': 'softs_cleaned_data.csv',
    'data': 'all_stocks_and_S&P_500',
    'features': 'M',
    'freq': 'd',
    'seq_len': 30,
    'pred_len': 14,
    # model settings
    'model': 'SOFTS',
    'checkpoints': './checkpoints/',
    'd_model': 128,
    'd_core': 64,
    'd_ff': 128,
    'e_layers': 2,
    'learning_rate': 0.0003,
    'lradj': 'cosine',
    'train_epochs': 50,
    'patience': 3,
    'batch_size': 16,
    'dropout': 0.0,
    'activation': 'gelu',
    'use_norm': True,
    # system settings
    'num_workers': 0,
    'use_gpu': True,
    'gpu': '0',
    'save_model': True,
}

parser = argparse.ArgumentParser(description='SOFTS')
args = parser.parse_args([])
args.__dict__.update(config)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)

#############################################################################################
# load data
print(args.root_path, args.data_path)
data = pd.read_csv(os.path.join(args.root_path, args.data_path))

if 'Date' in data.columns or 'date' in data.columns:
    date_col = 'Date' if 'Date' in data.columns else 'date'
    # Convert to datetime
    data[date_col] = pd.to_datetime(data[date_col])
    # Save dates before dropping for later use
    dates = data[date_col]
    data = data.drop(columns=[date_col])
print(data.head())

#split stock data
# total length
N = len(data)                   # 3019

# split points
train_end = int(N * 0.70)       # 70% train
val_end   = int(N * 0.85)       # next 15% val

#slice
train_data = data.iloc[: train_end]
vali_data  = data.iloc[train_end - args.seq_len : val_end]
test_data  = data.iloc[val_end - args.seq_len : ]

# optional: scale data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
if 'Date' in train_data.columns:
    scaler.fit(train_data.iloc[:, 1:])
    train_data.iloc[:, 1:] = scaler.transform(train_data.iloc[:, 1:])
    vali_data.iloc[:, 1:] = scaler.transform(vali_data.iloc[:, 1:])
    test_data.iloc[:, 1:] = scaler.transform(test_data.iloc[:, :])
else:
    scaler.fit(train_data.iloc[:, :])
    train_data.iloc[:, :] = scaler.transform(train_data.iloc[:, :])
    vali_data.iloc[:, :] = scaler.transform(vali_data.iloc[:, :])
    test_data.iloc[:, :] = scaler.transform(test_data.iloc[:, :])

#############################################################################################
# Add this before training
def preprocess_data(data):
    """
    Clean and preprocess data to prevent numerical issues
    """
    # Replace infinite values
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with column means
    data = data.fillna(data.mean())
    
    # Clip extreme values (winsorizing)
    for col in data.columns:
        q1 = data[col].quantile(0.01)
        q3 = data[col].quantile(0.99)
        data[col] = data[col].clip(q1, q3)
    
    return data

# Apply preprocessing
train_data = preprocess_data(train_data)
vali_data = preprocess_data(vali_data)
test_data = preprocess_data(test_data)

#############################################################################################
# Add this code cell before creating the Exp_Custom object
import types
import torch.nn.functional as F
from models.SOFTS import STAR

def patch_star_module(model):
    for module in model.modules():
        if hasattr(module, 'forward') and isinstance(module, STAR):
            original_forward = module.forward
            
            def safe_forward(self, input, *args, **kwargs):
                batch_size, channels, d_series = input.shape
                
                # Original code with minimal changes
                combined_mean = F.gelu(self.gen1(input))
                combined_mean = self.gen2(combined_mean)
                combined_mean = combined_mean - combined_mean.max(dim=1, keepdim=True)[0]
                
                if self.training:
                    # guardrails for torch.binomial
                    ratio = F.softmax(combined_mean, dim=1)
                    ratio = ratio.permute(0, 2, 1).reshape(-1, channels)
                    
                    # Fix for multinomial - ensure valid probabilities
                    ratio = torch.nan_to_num(ratio, nan=1e-10)
                    ratio = torch.abs(ratio) + 1e-10  # Ensure positive values
                    ratio = ratio / ratio.sum(dim=1, keepdim=True)  # Normalize
                    
                    indices = torch.multinomial(ratio, 1)
                    indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
                    combined_mean = torch.gather(combined_mean, 1, indices)
                    combined_mean = combined_mean.repeat(1, channels, 1)
                else:
                    # Keep inference code
                    weight = F.softmax(combined_mean, dim=1).clamp(min=1e-6)
                    weight = weight / weight.sum(dim=1, keepdim=True)
                    combined_mean = (combined_mean * weight).sum(dim=1, keepdim=True)
                    combined_mean = combined_mean.repeat(1, channels, 1)
                
                # Original fusion code
                combined_mean_cat = torch.cat([input, combined_mean], -1)
                combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
                combined_mean_cat = self.gen4(combined_mean_cat)
                
                return combined_mean_cat, None
                
            module.forward = types.MethodType(safe_forward, module)
    
    return model

#############################################################################################
# Train SOFTS Model
Exp = Exp_Custom(args)
patch_star_module(Exp.model)

setting = f'{args.data}_{args.model}_{args.seq_len}_{args.pred_len}'
print('>>>>>>>start training SOFTS : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
Exp.train(setting=setting, train_data=train_data, vali_data=vali_data, test_data=test_data)
print('>>>>>>>testing SOFTS : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
Exp.test(setting=setting, test_data=test_data)

# get SOFTS predictions
softs_predictions = Exp.predict(setting=setting, pred_data=test_data)
print("SOFTS predictions shape:", softs_predictions.shape)

#############################################################################################
# Train CNN Model
print('>>>>>>>start training CNN>>>>>>>>>>>>>>>>>>>>>>>>>>>')

# Prepare data for CNN
train_array = train_data.values
val_array = vali_data.values
test_array = test_data.values

# Create datasets
train_dataset = MultiStepDataset(train_array, args.seq_len, args.pred_len)
val_dataset = MultiStepDataset(val_array, args.seq_len, args.pred_len)
test_dataset = MultiStepDataset(test_array, args.seq_len, args.pred_len)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize CNN model
device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
cnn_model = EnhancedCNN1D(input_dim=len(data.columns), seq_len=args.seq_len, pred_len=args.pred_len).to(device)

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=args.learning_rate)

# Training loop
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(args.train_epochs):
    # Training phase
    cnn_model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = cnn_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation phase
    cnn_model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = cnn_model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    print(f'Epoch {epoch+1}/{args.train_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(cnn_model.state_dict(), f'./checkpoints/cnn_best_{setting}.pth')
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f'Early stopping after {epoch+1} epochs')
            break

# Load best model and make predictions
cnn_model.load_state_dict(torch.load(f'./checkpoints/cnn_best_{setting}.pth'))
cnn_model.eval()

cnn_predictions = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        outputs = cnn_model(batch_x)
        cnn_predictions.append(outputs.cpu().numpy())

cnn_predictions = np.concatenate(cnn_predictions, axis=0)
print("CNN predictions shape:", cnn_predictions.shape)

##############################################################################################
# Compare both models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Get actual values for SP500 (column index for SP500)
sp500_col_idx = list(data.columns).index('SP500')

# Extract predicted values for SP500 from both models
softs_predicted_sp500 = softs_predictions[:, :, sp500_col_idx]
cnn_predicted_sp500 = cnn_predictions[:, :, sp500_col_idx]

# Get actual test data for comparison
actual_sp500_values = []
for i in range(len(softs_predictions)):
    start_idx = args.seq_len + i
    if start_idx + args.pred_len <= len(test_data):
        actual_values = test_data.iloc[start_idx:start_idx+args.pred_len, sp500_col_idx].values
        actual_sp500_values.append(actual_values)

actual_sp500_array = np.array(actual_sp500_values)

# Make sure we're comparing same number of samples
min_samples = min(len(actual_sp500_array), len(softs_predicted_sp500), len(cnn_predicted_sp500))
actual_sp500_array = actual_sp500_array[:min_samples]
softs_predicted_sp500 = softs_predicted_sp500[:min_samples]
cnn_predicted_sp500 = cnn_predicted_sp500[:min_samples]

# Calculate metrics for both models
def calculate_metrics(actual, predicted, model_name):
    # First day metrics
    mae_first = mean_absolute_error(actual[:, 0], predicted[:, 0])
    rmse_first = np.sqrt(mean_squared_error(actual[:, 0], predicted[:, 0]))
    
    # All days metrics
    mae_all = mean_absolute_error(actual.flatten(), predicted.flatten())
    rmse_all = np.sqrt(mean_squared_error(actual.flatten(), predicted.flatten()))
    
    print(f"{model_name} - First day: MAE={mae_first:.4f}, RMSE={rmse_first:.4f}")
    print(f"{model_name} - All days: MAE={mae_all:.4f}, RMSE={rmse_all:.4f}")
    
    return mae_first, rmse_first, mae_all, rmse_all

softs_metrics = calculate_metrics(actual_sp500_array, softs_predicted_sp500, "SOFTS")
cnn_metrics = calculate_metrics(actual_sp500_array, cnn_predicted_sp500, "CNN")

# Inverse transform for plotting
def inverse_transform_sp500(values):
    temp_array = np.zeros((len(values), len(data.columns)))
    temp_array[:, sp500_col_idx] = values
    return scaler.inverse_transform(temp_array)[:, sp500_col_idx]

# Plot comparison of first day predictions
plt.figure(figsize=(14, 8))

actual_original = inverse_transform_sp500(actual_sp500_array[:, 0])
softs_original = inverse_transform_sp500(softs_predicted_sp500[:, 0])
cnn_original = inverse_transform_sp500(cnn_predicted_sp500[:, 0])

plt.subplot(2, 1, 1)
plt.plot(actual_original[5:], label='Actual SP500', linewidth=2)
plt.plot(softs_original[5:], label='SOFTS Predicted', linewidth=1.5, alpha=0.8)
plt.plot(cnn_original[5:], label='CNN Predicted', linewidth=1.5, alpha=0.8)
plt.title('SP500 - First Day Predictions Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 14-day windows comparison
plt.subplot(2, 1, 2)
step = 14
samples_to_plot = min(8, len(actual_sp500_array) // step)

for i in range(samples_to_plot):
    idx = i * step + 5
    if idx + args.pred_len <= len(actual_sp500_array):
        actual_window = inverse_transform_sp500(actual_sp500_array[idx, :])
        softs_window = inverse_transform_sp500(softs_predicted_sp500[idx, :])
        cnn_window = inverse_transform_sp500(cnn_predicted_sp500[idx, :])
        
        x_range = range(i*step, i*step+args.pred_len)
        plt.plot(x_range, actual_window, 'b-', alpha=0.7, linewidth=2)
        plt.plot(x_range, softs_window, 'r--', alpha=0.7, linewidth=1.5)
        plt.plot(x_range, cnn_window, 'g:', alpha=0.7, linewidth=1.5)

plt.title('SP500 - 14-Day Prediction Windows Comparison')
plt.xlabel('Days')
plt.ylabel('SP500 Price')
plt.grid(True, alpha=0.3)
# Add legend
plt.plot([], [], 'b-', label='Actual', linewidth=2)
plt.plot([], [], 'r--', label='SOFTS', linewidth=1.5)
plt.plot([], [], 'g:', label='CNN', linewidth=1.5)
plt.legend()

plt.tight_layout()
plt.savefig("model_comparison_SP500_predictions.png", dpi=300, bbox_inches='tight')
plt.show()

# Save individual model plots as well
plt.figure(figsize=(12, 6))
plt.plot(actual_original[5:], label='Actual SP500')
plt.plot(softs_original[5:], label='SOFTS Predicted')
plt.axvline(x=140, color='red', linestyle='dotted', linewidth=1)
plt.title('SP500 - SOFTS First Day Predictions vs Actual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("run-SP500-SOFTS-First-Day-Predictions-vs-Actual.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(actual_original[5:], label='Actual SP500')
plt.plot(cnn_original[5:], label='CNN Predicted')
plt.axvline(x=140, color='red', linestyle='dotted', linewidth=1)
plt.title('SP500 - CNN First Day Predictions vs Actual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("run-SP500-CNN-First-Day-Predictions-vs-Actual.png")
plt.show()

print("\nModel Comparison Summary:")
print("="*50)
print(f"SOFTS Model - First day MAE: {softs_metrics[0]:.4f}, RMSE: {softs_metrics[1]:.4f}")
print(f"CNN Model   - First day MAE: {cnn_metrics[0]:.4f}, RMSE: {cnn_metrics[1]:.4f}")
print(f"SOFTS Model - All days MAE: {softs_metrics[2]:.4f}, RMSE: {softs_metrics[3]:.4f}")
print(f"CNN Model   - All days MAE: {cnn_metrics[2]:.4f}, RMSE: {cnn_metrics[3]:.4f}")