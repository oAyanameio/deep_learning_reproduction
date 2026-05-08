# 导入PyTorch核心库
import torch
# 导入神经网络层模块
import torch.nn as nn
# 导入优化器相关
import torch.optim as optim
# 导入数据集与图像变换
from torchvision import datasets, transforms
# 导入数据加载器
from torch.utils.data import DataLoader
# 导入进度条库
from tqdm import tqdm
# 导入绘图库
import matplotlib.pyplot as plt

# 自动选择设备：GPU优先
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== CIFAR-10 数据预处理 ====================
transform = transforms.Compose([
    # 图像转张量
    transforms.ToTensor(),
    # 归一化到[-1,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR10训练集
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
# 加载CIFAR10测试集
test_dataset = datasets.CIFAR10(
    root="./data", train=False, transform=transform
)

# 训练集加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# 测试集加载器
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==================== 通用训练函数 ====================
def run_train(model, epochs=15, lr=1e-3):
    # 模型移到GPU/CPU
    model = model.to(device)
    # 分类任务用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 优化器使用Adam
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 记录损失和精度
    train_loss_list = []
    test_acc_list  = []

    for epoch in range(epochs):
        # 开启训练模式
        model.train()
        total_loss = 0

        # 遍历训练批次
        for data, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            data, label = data.to(device), label.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            out = model(data)
            # 计算损失
            loss = criterion(out, label)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            total_loss += loss.item()

        # 平均损失
        avg_loss = total_loss / len(train_loader)
        train_loss_list.append(avg_loss)

        # 测试阶段
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                out = model(data)
                pred = out.argmax(dim=1)
                correct += (pred == label).sum().item()

        acc = correct / len(test_dataset)
        test_acc_list.append(acc)
        print(f"Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    # 绘图
    plt.plot(train_loss_list, label="train loss")
    plt.plot(test_acc_list, label="test acc")
    plt.legend()
    plt.show()

    return model



class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 卷积+池化特征提取部分
        self.features = nn.Sequential(
            # Conv1: 3→6, 5x5卷积
            nn.Conv2d(3, 6, kernel_size=5),
            # 激活
            nn.ReLU(),
            # 池化 2x2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv2: 6→16
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

        )
        # 全连接分类部分
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平
        x = torch.flatten(x, 1)
        # 分类
        x = self.classifier(x)
        return x

# # 创建并训练
# model_lenet = LeNet5()
# run_train(model_lenet, epochs=15)、

class AlexNet(nn.Module):
    """
    AlexNet卷积神经网络模型（适配CIFAR-10数据集）
    
    该模型包含特征提取器和分类器两部分：
    - 特征提取器：4个卷积层，逐步提取图像特征并降低空间维度
    - 分类器：全连接层网络，将特征映射到类别概率
    
    Args:
        num_classes (int): 分类类别数量，默认为10（对应CIFAR-10的10个类别）
    
    Attributes:
        features (nn.Sequential): 特征提取网络，包含卷积、激活和池化层
        classifier (nn.Sequential): 分类网络，包含Dropout和全连接层
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # Conv4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            # Dropout 防止过拟合
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# model_alex = AlexNet()
# run_train(model_alex, epochs=15)

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 块1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 块2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 块3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# model_vgg = VGG16()
# run_train(model_vgg, epochs=15)

# 定义残差块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主分支卷积层1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        # 批归一化
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 激活
        self.relu = nn.ReLU(inplace=True)
        # 卷积层2
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # shortcut 旁路（维度不一致时使用）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主分支计算
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差相加：主分支 + shortcut
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# 定义ResNet18
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 32

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        # 堆叠残差块
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 分类层
        self.fc = nn.Linear(128, num_classes)

    # 自动堆叠残差块
    def _make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(ResBlock(self.in_channels, channels, s))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

model_res = ResNet18()
run_train(model_res, epochs=15)