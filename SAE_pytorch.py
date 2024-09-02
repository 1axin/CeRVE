# 作者:     wxf

# 开发时间: 2023/11/21 11:07
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


class SparseAutoencoder(nn.Module) :
    def __init__(self, input_size, hidden_size, sparsity_factor) :
        super(SparseAutoencoder, self).__init__()

        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.sparsity_factor = sparsity_factor
        self.sparsity_target = 0.1

    def forward(self, x) :
        encoded = self.sigmoid(self.encoder(x))
        decoded = self.sigmoid(self.decoder(encoded))

        return encoded, decoded

    def compute_sparsity_loss(self, encoded) :
        avg_activation = torch.mean(encoded, dim = 0)
        sparsity_loss = self.sparsity_factor * torch.mean(
            self.kl_divergence(self.sparsity_target, avg_activation)
        )

        return sparsity_loss

    def kl_divergence(self, p, q) :
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, )
        writer.writerows(data)
    return

# 示例用法
import pandas as pd
import torch

# 从 CSV 文件中读取特征数据
df = pd.read_csv('相似性特征-miRNA-CMI-LMI-MGA-MDA-SampleFeature-无序号.csv', header=None)

# 获取特征部分（假设特征在第 1 到倒数第 1 列之间）
features = df.iloc[:].values

# # 进行数据标准化（可按照具体需求进行调整）
# normalized_features = (features - features.mean()) / features.std()

# 转换为 PyTorch Tensor
inputs = torch.tensor(features, dtype=torch.float32)

input_size = 1401  # 输入大小
hidden_size = 64  # 隐层大小
sparsity_factor = 0.1  # 稀疏因子
num_epochs = 10  # 训练迭代次数

# 创建稀疏自编码器模型
model = SparseAutoencoder(input_size, hidden_size, sparsity_factor)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.MSELoss()

# 进行训练
for epoch in range(num_epochs) :
    # 前向传播
    encoded, decoded = model(inputs)

    # 计算重构损失
    reconstruction_loss = criterion(decoded, inputs)

    # 计算稀疏性损失
    sparsity_loss = model.compute_sparsity_loss(encoded)

    # 总损失
    loss = reconstruction_loss + sparsity_loss

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

# 前向传播获取编码器的输出
with torch.no_grad() :
    encoded, _ = model(inputs)

# 转换为 numpy 数组
encoded_numpy = encoded.numpy()

# 将特征输出保存到 CSV 文件
df = pd.DataFrame(encoded_numpy)
df = df.values

data1 = []
ReadMyCsv(data1, "568-miRNA-序号.csv")
print(len(data1))

data = []
data = np.hstack((data1,df))

# data.to_csv('72-CANCER-SAE-64-序号.csv', index = False, header=None)
storFile(data, "568-miRNA-SAE-64-序号.csv")
print(len(data))