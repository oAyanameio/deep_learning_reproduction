# 导入PyTorch基础库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入优化器模块
import torch.optim as optim
# 导入数据集和图像处理工具
from torchvision import datasets, transforms
# 导入数据加载器工具
from torch.utils.data import DataLoader
# 导入进度条工具
from tqdm import tqdm
# 导入绘图库
import matplotlib.pyplot as plt

# 选择设备：有GPU用GPU，没有用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像预处理流程：转张量 + 标准化
transform = transforms.Compose([
    # 将图片转换为张量
    transforms.ToTensor(),
    # MNIST数据标准化（固定均值和方差）
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST训练集
train_dataset = datasets.MNIST(
    # 数据保存路径
    root='./data',
    # 训练集
    train=True,
    # 不存在则自动下载
    download=True,
    # 应用预处理
    transform=transform
)

# 加载MNIST测试集
test_dataset = datasets.MNIST(
    # 数据路径
    root='./data',
    # 测试集
    train=False,
    # 应用预处理
    transform=transform
)

# 训练数据加载器：批量大小64，随机打乱
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# 测试数据加载器：批量大小64，不打乱
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 通用训练函数：传入模型、训练轮数、学习率
def run_train(model, epochs=10, lr=1e-3):
    # 将模型移动到指定设备
    model = model.to(device)
    # 定义损失函数：交叉熵损失（分类任务专用）
    criterion = nn.CrossEntropyLoss()
    # 定义优化器：Adam优化模型参数
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 记录每轮训练损失
    train_loss_list = []
    # 记录每轮测试精度
    test_acc_list = []

    # 开始循环训练轮数
    for epoch in range(epochs):
        # 设置模型为训练模式
        model.train()
        # 累计本轮总损失
        total_loss = 0

        # 遍历训练数据（带进度条）
        for data, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # 数据移到设备
            data, label = data.to(device), label.to(device)
            # 清空上一步梯度
            optimizer.zero_grad()
            # 前向传播：模型输出预测值
            out = model(data)
            # 计算预测值与真实标签的损失
            loss = criterion(out, label)
            # 反向传播：计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 累计损失值
            total_loss += loss.item()

        # 计算本轮平均损失
        avg_loss = total_loss / len(train_loader)
        # 保存损失到列表
        train_loss_list.append(avg_loss)

        # 设置模型为评估模式
        model.eval()
        # 记录预测正确的数量
        correct = 0

        # 测试阶段不计算梯度
        with torch.no_grad():
            # 遍历测试集
            for data, label in test_loader:
                # 数据移到设备
                data, label = data.to(device), label.to(device)
                # 前向传播
                out = model(data)
                # 取输出最大值作为预测类别
                pred = out.argmax(dim=1)
                # 统计正确预测数量
                correct += (pred == label).sum().item()

        # 计算测试准确率
        acc = correct / len(test_dataset)
        # 保存准确率
        test_acc_list.append(acc)
        # 打印本轮结果
        print(f"Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    # 绘制损失和准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.title('Training Loss and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存到 training_curves.png")
    plt.show()
    return model

# 定义单层感知机模型
class LinearModel(nn.Module):
    # 构造函数
    def __init__(self):
        # 调用父类构造函数
        super().__init__()
        # 定义展平层：将28*28图像展平为一维向量
        self.flatten = nn.Flatten()
        # 定义线性层：输入784维，输出10维（对应10个数字）
        self.linear = nn.Linear(28*28, 10)

    # 前向传播函数
    def forward(self, x):
        # 将输入展平
        x = self.flatten(x)
        # 线性变换得到输出
        x = self.linear(x)
        # 返回输出
        return x
# # 创建模型实例
# model1 = LinearModel()
# # 开始训练
# run_train(model1, epochs=10)

# 定义多层感知机模型
class MLP(nn.Module):
    # 构造函数
    def __init__(self):
        # 调用父类构造
        super().__init__()
        # 展平层
        self.flatten = nn.Flatten()
        # 第一层线性层：784→256
        self.fc1 = nn.Linear(28*28, 256)
        # 第一层激活函数ReLU
        self.relu1 = nn.ReLU()
        # 第二层线性层：256→128
        self.fc2 = nn.Linear(256, 128)
        # 第二层激活函数ReLU
        self.relu2 = nn.ReLU()
        # 输出层：128→10
        self.fc3 = nn.Linear(128, 10)

    # 前向传播
    def forward(self, x):
        # 展平输入
        print("输入形状:", x.shape)
        print("输入前3个样本的前5个元素:", x[:3, :, 0, :5])
        x = self.flatten(x)
        print("展平后形状:", x.shape)
        print("展平后前3个样本的前10个元素:", x[:3, :10])
        # 第一层线性变换
        x = self.fc1(x)
        print("第一层线性变换后形状:", x.shape)
        print("第一层线性变换后前3个样本的前5个元素:", x[:3, :5])
        # 第一层非线性激活
        x = self.relu1(x)
        print("第一层ReLU激活后形状:", x.shape)
        print("第一层ReLU激活后前3个样本的前5个元素:", x[:3, :5])
        # 第二层线性变换
        x = self.fc2(x)
        print("第二层线性变换后形状:", x.shape)
        print("第二层线性变换后前3个样本的前5个元素:", x[:3, :5])
        # 第二层非线性激活
        x = self.relu2(x)
        print("第二层ReLU激活后形状:", x.shape)
        print("第二层ReLU激活后前3个样本的前5个元素:", x[:3, :5])
        # 输出层
        x = self.fc3(x)
        print("输出层形状:", x.shape)
        print("输出层前3个样本:", x[:3])
        # 返回结果
        return x

# 创建模型
model2 = MLP()
# 训练
run_train(model2, epochs=10)

