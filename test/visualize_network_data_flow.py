import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

class MLPWithActivationCapture(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
        
        self.activations = {}

    def forward(self, x):
        self.activations['input'] = x.detach().cpu().numpy()
        x = self.flatten(x)
        self.activations['flatten'] = x.detach().cpu().numpy()
        
        x = self.fc1(x)
        self.activations['fc1'] = x.detach().cpu().numpy()
        x = self.relu1(x)
        self.activations['relu1'] = x.detach().cpu().numpy()
        
        x = self.fc2(x)
        self.activations['fc2'] = x.detach().cpu().numpy()
        x = self.relu2(x)
        self.activations['relu2'] = x.detach().cpu().numpy()
        
        x = self.fc3(x)
        self.activations['output'] = x.detach().cpu().numpy()
        
        return x

def plot_activation_maps(activations, sample_idx=0):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    ax_idx = 0
    axes[ax_idx].imshow(activations['input'][sample_idx, 0], cmap='gray')
    axes[ax_idx].set_title('Input Image\n(28x28)')
    axes[ax_idx].axis('off')
    ax_idx += 1
    
    axes[ax_idx].hist(activations['flatten'][sample_idx], bins=50, color='skyblue')
    axes[ax_idx].set_title('Flatten Layer\n(784 features)')
    axes[ax_idx].set_xlabel('Activation Value')
    axes[ax_idx].set_ylabel('Count')
    ax_idx += 1
    
    axes[ax_idx].hist(activations['fc1'][sample_idx], bins=50, color='orange')
    axes[ax_idx].set_title('FC1 Layer\n(256 features)')
    axes[ax_idx].set_xlabel('Activation Value')
    axes[ax_idx].set_ylabel('Count')
    ax_idx += 1
    
    axes[ax_idx].hist(activations['relu1'][sample_idx], bins=50, color='green')
    axes[ax_idx].set_title('ReLU1 Activation\n(256 features)')
    axes[ax_idx].set_xlabel('Activation Value')
    axes[ax_idx].set_ylabel('Count')
    ax_idx += 1
    
    axes[ax_idx].hist(activations['fc2'][sample_idx], bins=50, color='purple')
    axes[ax_idx].set_title('FC2 Layer\n(128 features)')
    axes[ax_idx].set_xlabel('Activation Value')
    axes[ax_idx].set_ylabel('Count')
    ax_idx += 1
    
    axes[ax_idx].hist(activations['relu2'][sample_idx], bins=50, color='red')
    axes[ax_idx].set_title('ReLU2 Activation\n(128 features)')
    axes[ax_idx].set_xlabel('Activation Value')
    axes[ax_idx].set_ylabel('Count')
    ax_idx += 1
    
    output_probs = nn.functional.softmax(torch.tensor(activations['output'][sample_idx]), dim=0).numpy()
    axes[ax_idx].bar(range(10), output_probs, color='blue')
    axes[ax_idx].set_title('Output Layer\n(10 classes)')
    axes[ax_idx].set_xlabel('Digit Class')
    axes[ax_idx].set_ylabel('Probability')
    axes[ax_idx].set_xticks(range(10))
    ax_idx += 1
    
    axes[ax_idx].imshow(activations['relu1'][sample_idx].reshape(16, 16), cmap='viridis')
    axes[ax_idx].set_title('ReLU1 Feature Map\n(16x16)')
    axes[ax_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/lbh/deep_learning_reproduction/test/network_data_flow.png', dpi=300, bbox_inches='tight')
    print("可视化图像已保存到: /home/lbh/deep_learning_reproduction/test/network_data_flow.png")
    plt.show()

def print_layer_statistics(activations, sample_idx=0):
    print("="*70)
    print("神经网络各层数据变化统计")
    print("="*70)
    
    layers = [
        ('输入层 (Input)', 'input', '图像 28x28'),
        ('展平层 (Flatten)', 'flatten', '向量 784'),
        ('全连接层1 (FC1)', 'fc1', '向量 256'),
        ('ReLU激活1', 'relu1', '向量 256'),
        ('全连接层2 (FC2)', 'fc2', '向量 128'),
        ('ReLU激活2', 'relu2', '向量 128'),
        ('输出层 (Output)', 'output', '向量 10')
    ]
    
    for layer_name, key, shape_desc in layers:
        data = activations[key][sample_idx]
        print(f"\n{layer_name}: {shape_desc}")
        print(f"  最小值: {np.min(data):.4f}")
        print(f"  最大值: {np.max(data):.4f}")
        print(f"  平均值: {np.mean(data):.4f}")
        print(f"  标准差: {np.std(data):.4f}")
        if key == 'output':
            probs = nn.functional.softmax(torch.tensor(data), dim=0).numpy()
            print(f"  预测类别: {np.argmax(probs)} (概率: {np.max(probs):.4f})")

def main():
    print("加载预训练模型...")
    model = MLPWithActivationCapture().to(device)
    
    print("加载测试数据...")
    sample_data, sample_label = test_dataset[42]
    sample_data = sample_data.unsqueeze(0).to(device)
    
    print("\n执行前向传播...")
    with torch.no_grad():
        output = model(sample_data)
    
    print("\n真实标签:", sample_label)
    print("预测结果:", output.argmax(dim=1).item())
    
    print_layer_statistics(model.activations)
    plot_activation_maps(model.activations)

if __name__ == '__main__':
    main()