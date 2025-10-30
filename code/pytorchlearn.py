"""
PyTorch学习示例：拟合函数 y = x1^2 + x2^2
完整演示数据构建、模型构建、损失构建、优化器构建、训练与评估过程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========== 1. 数据构建 ==========
class QuadraticDataset(Dataset):
    """自定义数据集类，生成 y = x1^2 + x2^2 的数据"""
    
    def __init__(self, num_samples=10000, x_range=(-5, 5), noise_std=0.1):
        """
        Args:
            num_samples: 生成样本数量
            x_range: x1, x2的取值范围
            noise_std: 噪声标准差
        """
        self.num_samples = num_samples
        
        # 生成随机输入数据 x1, x2
        self.x1 = np.random.uniform(x_range[0], x_range[1], num_samples)
        self.x2 = np.random.uniform(x_range[0], x_range[1], num_samples)
        
        # 计算真实标签 y = x1^2 + x2^2
        self.y = self.x1**2 + self.x2**2
        
        # 添加一些噪声使数据更真实
        noise = np.random.normal(0, noise_std, num_samples)
        self.y += noise
        
        # 转换为torch tensor
        self.features = torch.FloatTensor(np.column_stack([self.x1, self.x2]))
        self.targets = torch.FloatTensor(self.y)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 创建训练和测试数据集
print("正在生成数据...")
train_dataset = QuadraticDataset(num_samples=8000, noise_std=0.1)
test_dataset = QuadraticDataset(num_samples=2000, noise_std=0.1)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"批次大小: {batch_size}")

# 可视化部分数据
plt.figure(figsize=(15, 5))

# 绘制3D散点图
from mpl_toolkits.mplot3d import Axes3D

# 选择前1000个点进行可视化
sample_size = 1000
sample_x1 = train_dataset.x1[:sample_size]
sample_x2 = train_dataset.x2[:sample_size]
sample_y = train_dataset.y[:sample_size]

ax1 = plt.subplot(1, 3, 1, projection='3d')
ax1.scatter(sample_x1, sample_x2, sample_y, alpha=0.6, s=1)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y = x1² + x2²')
ax1.set_title('3D数据分布')

# 绘制x1与y的关系
plt.subplot(1, 3, 2)
plt.scatter(sample_x1, sample_y, alpha=0.6, s=1)
plt.xlabel('x1')
plt.ylabel('y')
plt.title('x1 vs y')
plt.grid(True)

# 绘制x2与y的关系
plt.subplot(1, 3, 3)
plt.scatter(sample_x2, sample_y, alpha=0.6, s=1)
plt.xlabel('x2')
plt.ylabel('y')
plt.title('x2 vs y')
plt.grid(True)

plt.tight_layout()
plt.show()

# ========== 2. 模型构建 ==========
class QuadraticNet(nn.Module):
    """神经网络模型用于拟合二次函数"""
    
    def __init__(self, input_dim=2, hidden_dims=[64, 32, 16], output_dim=1):
        """
        Args:
            input_dim: 输入维度 (x1, x2)
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度 (y)
        """
        super(QuadraticNet, self).__init__()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 添加dropout防止过拟合
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 组合所有层
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# 创建模型实例
model = QuadraticNet(input_dim=2, hidden_dims=[64, 32, 16], output_dim=1)
model = model.to(device)

# 打印模型结构
print("\n模型结构:")
print(model)

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")

# ========== 3. 损失函数构建 ==========
# 使用均方误差损失
criterion = nn.MSELoss()

# 也可以尝试其他损失函数
# criterion = nn.L1Loss()  # 平均绝对误差
# criterion = nn.SmoothL1Loss()  # Huber损失

print(f"\n损失函数: {criterion}")

# ========== 4. 优化器构建 ==========
# 使用Adam优化器
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.8, patience=10
)

# 也可以尝试其他优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001)

print(f"优化器: {optimizer}")
print(f"学习率调度器: {scheduler}")

# ========== 5. 训练过程 ==========
def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # 前向传播
        predictions = model(batch_x).squeeze()
        loss = criterion(predictions, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions = model(batch_x).squeeze()
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    # 计算R²分数
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    return avg_loss, r2_score, all_predictions, all_targets

# 训练参数
num_epochs = 100
train_losses = []
test_losses = []
r2_scores = []

print(f"\n开始训练...")
print(f"训练epochs: {num_epochs}")
print(f"学习率: {learning_rate}")

best_test_loss = float('inf')
best_model_state = None
previous_lr = learning_rate

# 训练循环
for epoch in range(num_epochs):
    # 训练
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # 评估
    test_loss, r2_score, _, _ = evaluate(model, test_loader, criterion, device)
    
    # 记录损失
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    r2_scores.append(r2_score)
    
    # 学习率调度
    scheduler.step(test_loss)
    
    # 检查学习率是否变化
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr != previous_lr:
        print(f'学习率从 {previous_lr:.8f} 降低到 {current_lr:.8f}')
        previous_lr = current_lr
    
    # 保存最佳模型
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_state = model.state_dict().copy()
    
    # 每10个epoch打印一次进度
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  训练损失: {train_loss:.6f}')
        print(f'  测试损失: {test_loss:.6f}')
        print(f'  R² 分数: {r2_score:.6f}')
        print(f'  学习率: {current_lr:.8f}')
        print()

print("训练完成!")

# 加载最佳模型
model.load_state_dict(best_model_state)
print(f"已加载最佳模型 (测试损失: {best_test_loss:.6f})")

# ========== 6. 结果评估与可视化 ==========

# 最终评估
final_test_loss, final_r2, predictions, targets = evaluate(model, test_loader, criterion, device)

print(f"\n最终评估结果:")
print(f"测试损失 (MSE): {final_test_loss:.6f}")
print(f"R² 分数: {final_r2:.6f}")
print(f"均方根误差 (RMSE): {np.sqrt(final_test_loss):.6f}")

# 绘制训练曲线
plt.figure(figsize=(15, 10))

# 损失曲线
plt.subplot(2, 3, 1)
plt.plot(train_losses, label='训练损失', alpha=0.8)
plt.plot(test_losses, label='测试损失', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('训练 vs 测试损失')
plt.legend()
plt.grid(True)

# R²分数曲线
plt.subplot(2, 3, 2)
plt.plot(r2_scores, label='R² 分数', color='green', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('R² 分数变化')
plt.legend()
plt.grid(True)

# 预测 vs 真实值散点图
plt.subplot(2, 3, 3)
plt.scatter(targets, predictions, alpha=0.6, s=1)
plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('预测 vs 真实值')
plt.grid(True)

# 残差图
plt.subplot(2, 3, 4)
residuals = targets - predictions
plt.scatter(predictions, residuals, alpha=0.6, s=1)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差分析')
plt.grid(True)

# 残差直方图
plt.subplot(2, 3, 5)
plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('残差')
plt.ylabel('频次')
plt.title('残差分布')
plt.grid(True)

# 预测误差分布
plt.subplot(2, 3, 6)
errors = np.abs(residuals)
plt.hist(errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
plt.xlabel('绝对误差')
plt.ylabel('频次')
plt.title('绝对误差分布')
plt.grid(True)

plt.tight_layout()
plt.show()

# ========== 7. 模型测试示例 ==========
print("\n========== 模型测试示例 ==========")

# 测试一些具体的点
test_points = [
    [0, 0],      # 期望输出: 0
    [1, 1],      # 期望输出: 2
    [2, 3],      # 期望输出: 13
    [-1, 2],     # 期望输出: 5
    [3, -2],     # 期望输出: 13
]

model.eval()
print("测试点预测:")
print("输入 (x1, x2) -> 预测值 | 真实值 | 误差")
print("-" * 40)

with torch.no_grad():
    for x1, x2 in test_points:
        input_tensor = torch.FloatTensor([[x1, x2]]).to(device)
        prediction = model(input_tensor).item()
        true_value = x1**2 + x2**2
        error = abs(prediction - true_value)
        
        print(f"({x1:2}, {x2:2}) -> {prediction:8.4f} | {true_value:8.4f} | {error:6.4f}")

# ========== 8. 保存模型 ==========
model_save_path = "quadratic_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_losses': test_losses,
    'r2_scores': r2_scores,
    'best_test_loss': best_test_loss,
    'model_config': {
        'input_dim': 2,
        'hidden_dims': [64, 32, 16],
        'output_dim': 1
    }
}, model_save_path)

print(f"\n模型已保存到: {model_save_path}")

# 加载模型示例
def load_model(model_path):
    """加载保存的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 重新创建模型
    config = checkpoint['model_config']
    loaded_model = QuadraticNet(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        output_dim=config['output_dim']
    ).to(device)
    
    # 加载权重
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"模型已从 {model_path} 加载")
    print(f"最佳测试损失: {checkpoint['best_test_loss']:.6f}")
    
    return loaded_model, checkpoint

# 演示加载模型
print("\n演示模型加载:")
loaded_model, checkpoint = load_model(model_save_path)

print("\n========== PyTorch学习完成 ==========")
print("本示例涵盖了以下内容:")
print("1. ✅ 数据构建 - 自定义Dataset类生成二次函数数据")
print("2. ✅ 模型构建 - 多层神经网络with ReLU和Dropout")
print("3. ✅ 损失构建 - MSE损失函数")
print("4. ✅ 优化器构建 - Adam优化器 + 学习率调度")
print("5. ✅ 训练过程 - 完整的训练循环with最佳模型保存")
print("6. ✅ 评估分析 - 损失曲线、R²分数、残差分析")
print("7. ✅ 模型保存与加载 - checkpoint机制")
print("8. ✅ 实际预测测试 - 具体数值验证")
