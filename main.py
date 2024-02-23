import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
# 加载数据
file_path = 'D:\korean\Metal_1.csv'
data = pd.read_csv(file_path)
data['Time'] = pd.to_datetime(data['Time'])
data.set_index('Time', inplace=True)

# 数据规范化
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data['Power consumption'].values.reshape(-1, 1))
data_scaled = data_scaled[:50000]
# 定义数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, series, window_size):
        self.series = series
        self.window_size = window_size

    def __len__(self):
        return len(self.series) - self.window_size

    def __getitem__(self, index):
        return (self.series[index:index+self.window_size], self.series[index+self.window_size])

# 参数设置
window_size = 60
batch_size = 32
n_features = 1

# 划分数据集
X = data_scaled
train_data, test_data = train_test_split(X, test_size=0.2, shuffle=False)
train_dataset = TimeSeriesDataset(train_data, window_size)
test_dataset = TimeSeriesDataset(test_data, window_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, dim_model, num_heads, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.window_size = window_size
        self.linear_in = nn.Linear(input_dim, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_out = nn.Linear(dim_model * window_size, 1)

    def forward(self, src):
        src = self.linear_in(src)
        output = self.transformer_encoder(src)
        output = output.reshape(output.size(0), -1)
        output = self.final_out(output)
        return output

# 实例化模型
model = TransformerModel(input_dim=n_features, dim_model=512, num_heads=8, num_layers=3, dropout=0.1).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 训练模型
def train(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for seq, labels in train_loader:
            seq = seq.float()
            labels = labels.float()
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

train(model, train_loader, criterion, optimizer)

# 测试模型
def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq, labels in test_loader:
            seq = seq.float()
            labels = labels.float()
            seq, labels = seq.to(device), labels.to(device)
            output = model(seq)
            loss = criterion(output, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')

evaluate(model, test_loader)
