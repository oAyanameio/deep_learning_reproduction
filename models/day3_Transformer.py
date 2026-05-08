import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 数据集预处理（适配 Transformer 的 32x32 输入）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def run_train(model, epochs=10, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_list = []
    test_acc_list = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_loss_list.append(avg_loss)

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

    return model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        # 随机失活层
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 生成位置下标：0,1,2...
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 三角函数频率分母
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数位 sin 编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位 cos 编码
        pe[:, 1::2] = torch.cos(position * div_term)
        # 适配批次维度
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 注册为无需梯度的固定参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 给序列叠加位置编码
        x = x + self.pe[:x.size(0), :]
        # 随机丢弃
        return self.dropout(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True     # 批次维度在前，统一格式
        )

        # 前馈神经网络隐藏层
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # 激活函数
        self.activation = nn.GELU()
        # 输出映射层
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 两层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 两个dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # ========== 多头自注意力 + 残差 ==========
        # 第一次层归一化
        x_norm = self.norm1(x)
        # 自注意力计算 Q=K=V
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        # 残差连接 + dropout
        x = x + self.dropout1(attn_out)

        # ========== FFN前馈网络 + 残差 ==========
        # 第二次层归一化
        x_norm2 = self.norm2(x)
        # 两层全连接 + 激活
        ffn_out = self.linear2(self.activation(self.linear1(x_norm2)))
        # 残差连接
        x = x + self.dropout2(ffn_out)

        return x

class FullTransformer(nn.Module):
    def __init__(self,
                 img_size=32,        # 输入图像尺寸
                 patch_size=4,       # 图像分块大小
                 in_c=3,             # 输入通道数
                 d_model=128,        # 模型总维度
                 nhead=8,            # 注意力头数
                 num_layers=6,       # Transformer层数
                 dim_feedforward=256,# 前馈网络维度
                 num_classes=10,    # 分类类别数
                 dropout=0.1):
        super().__init__()

        # 计算单个方向patch数量
        self.num_patches = (img_size // patch_size) ** 2

        # Patch嵌入：将图像小块映射为向量
        self.patch_embed = nn.Conv2d(
            in_channels=in_c,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 堆叠多层Transformer Block
        self.encoder_layers = nn.Sequential(
            *[TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
              for _ in range(num_layers)]
        )

        # 全局归一化
        self.norm = nn.LayerNorm(d_model)
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)

    def _print_tensor_content(self, x, max_elements=20):
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 4:
            print("    张量切片 [0, :, :2, :2]（第1张图的前2x2像素）:")
            for ch in range(min(x_np.shape[1], 3)):
                print(f"      通道{ch}:")
                for row in x_np[0, ch, :2, :2]:
                    print(f"        {['%.3f' % v for v in row]}")
        elif x_np.ndim == 3:
            print("    张量切片 [0, :3, :5]（第1个样本的前3个token的前5个维度）:")
            for i in range(min(x_np.shape[1], 3)):
                print(f"      token[{i}]: {['%.3f' % v for v in x_np[0, i, :5]]}")
        elif x_np.ndim == 2:
            print("    张量切片 [:3, :5]（前3行前5列）:")
            for i in range(min(x_np.shape[0], 3)):
                print(f"      {['%.3f' % v for v in x_np[i, :5]]}")
        else:
            print(f"    张量前{min(len(x_np), max_elements)}个值:")
            print(f"      {['%.3f' % v for v in x_np.flatten()[:max_elements]]}")

    def _print_stats(self, x, name):
        x_cpu = x.detach().cpu()
        print(f"┌─────────────────────────────────────────────────────────────")
        print(f"│ [{name}]")
        print(f"├─────────────────────────────────────────────────────────────")
        print(f"│ 形状: {tuple(x.shape)}")
        print(f"│ 维度数: {x.dim()}")
        print(f"│ 元素总数: {x.numel()}")
        print(f"├─────────────────────────────────────────────────────────────")
        print(f"│ 均值: {x_cpu.mean().item():.6f}")
        print(f"│ 标准差: {x_cpu.std().item():.6f}")
        print(f"│ 最小值: {x_cpu.min().item():.6f}")
        print(f"│ 最大值: {x_cpu.max().item():.6f}")
        print(f"│ 非零比例: {(x_cpu != 0).float().mean().item():.6f}")
        print(f"│ 是否包含NaN: {torch.any(torch.isnan(x_cpu)).item()}")
        print(f"│ 是否包含Inf: {torch.any(torch.isinf(x_cpu)).item()}")
        print(f"├─────────────────────────────────────────────────────────────")
        print(f"│ 张量内容预览:")
        self._print_tensor_content(x)
        print(f"└─────────────────────────────────────────────────────────────")
        print()

    def forward(self, x):
        self._print_stats(x, "输入")
        # 1. Patch嵌入：[B,3,32,32] -> [B,d_model,H,W]
        x = self.patch_embed(x)
        self._print_stats(x, "Patch嵌入后")
        # 2. 展平为序列：[B,d_model,N_patch] -> [B,N_patch,d_model]
        x = x.flatten(2).transpose(1, 2)
        self._print_stats(x, "展平为序列后")
        # 3. 加入位置编码
        x = self.pos_encoder(x)
        self._print_stats(x, "加入位置编码后")
        # 4. 多层Transformer编码
        x = self.encoder_layers(x)
        self._print_stats(x, "Transformer编码后")
        # 5. 全局均值池化，聚合全局特征
        x = x.mean(dim=1)
        self._print_stats(x, "均值池化后")
        # 6. 归一化 + 分类
        x = self.norm(x)
        self._print_stats(x, "归一化后")
        out = self.classifier(x)
        self._print_stats(out, "分类输出")
        return out

if __name__ == '__main__':
    # 初始化完整Transformer
    model_transformer = FullTransformer(
        img_size=32,
        patch_size=4,
        in_c=3,
        d_model=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=256,
        num_classes=10
    )

    # 开始训练（只运行1轮）
    run_train(model_transformer, epochs=1)