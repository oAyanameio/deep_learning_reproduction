import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import inspect

class NetworkTester:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.activations = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Flatten, nn.Softmax)):
                hook = module.register_forward_hook(self._save_activation(name))
                self.hooks.append(hook)
    
    def _save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu().numpy()
        return hook
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def test_forward_pass(self, sample_data):
        self.activations = {}
        sample_data = sample_data.to(self.device)
        with torch.no_grad():
            output = self.model(sample_data)
        return output, self.activations
    
    def test_training(self, train_loader, criterion=None, optimizer=None, epochs=3, lr=1e-3):
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"\n{'='*60}")
        print(f"开始训练测试 - 共 {epochs} 轮")
        print(f"{'='*60}")
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for data, labels in progress_bar:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f"第 {epoch+1} 轮 - 平均损失: {avg_loss:.4f}")
        
        print(f"\n{'='*60}")
        print("训练测试完成！")
        print(f"{'='*60}")
        return avg_loss
    
    def test_inference(self, test_loader):
        print(f"\n{'='*60}")
        print("开始推理测试")
        print(f"{'='*60}")
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in tqdm(test_loader, desc="推理中"):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"推理准确率: {accuracy:.4f} ({correct}/{total})")
        print(f"{'='*60}")
        return accuracy
    
    def visualize_data_flow(self, sample_data, save_path=None):
        output, activations = self.test_forward_pass(sample_data)
        
        print(f"\n{'='*60}")
        print("数据流动可视化 - 各层统计信息")
        print(f"{'='*60}")
        
        layer_info = []
        for layer_name, activation in activations.items():
            shape = activation.shape
            layer_info.append({
                'name': layer_name,
                'shape': shape,
                'min': np.min(activation),
                'max': np.max(activation),
                'mean': np.mean(activation),
                'std': np.std(activation),
                'data': activation
            })
            print(f"层: {layer_name}")
            print(f"  形状: {shape}")
            print(f"  最小值: {np.min(activation):.4f}")
            print(f"  最大值: {np.max(activation):.4f}")
            print(f"  平均值: {np.mean(activation):.4f}")
            print(f"  标准差: {np.std(activation):.4f}")
            print()
        
        self._plot_activations(layer_info, save_path)
        return layer_info
    
    def _plot_activations(self, layer_info, save_path=None):
        num_plots = min(len(layer_info), 8)
        rows = (num_plots + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(20, rows * 5))
        axes = axes.flatten()
        
        for i, info in enumerate(layer_info[:num_plots]):
            ax = axes[i]
            data = info['data'][0]
            
            if len(data.shape) == 3:
                data = data.transpose(1, 2, 0) if data.shape[0] < data.shape[1] else data[0]
            elif len(data.shape) == 2:
                if data.shape[0] == data.shape[1]:
                    pass
                else:
                    data = data[:min(256, len(data))].reshape(int(np.sqrt(min(256, len(data)))), -1)
            elif len(data.shape) == 1:
                ax.hist(data, bins=50, color='skyblue')
                ax.set_title(f"{info['name']}\n{info['shape']}")
                ax.set_xlabel('激活值')
                ax.set_ylabel('数量')
                continue
            
            if data.ndim == 2:
                im = ax.imshow(data, cmap='viridis')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.hist(data.flatten(), bins=50, color='skyblue')
                ax.set_xlabel('激活值')
                ax.set_ylabel('数量')
            
            ax.set_title(f"{info['name']}\n{info['shape']}")
            ax.axis('off')
        
        for i in range(num_plots, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化图像已保存到: {save_path}")
        plt.close()
    
    def summary(self, input_size):
        print(f"\n{'='*60}")
        print("模型结构摘要")
        print(f"{'='*60}")
        
        input_tensor = torch.randn(*input_size).to(self.device)
        self.activations = {}
        
        with torch.no_grad():
            self.model(input_tensor)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print()
        print("各层输出形状:")
        print("-" * 60)
        
        for name, activation in self.activations.items():
            print(f"{name}: {activation.shape}")
        
        print(f"{'='*60}")
        return total_params, trainable_params

def get_default_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def run_full_test(model, model_name="Unknown", epochs=3, batch_size=64):
    print(f"\n{'='*70}")
    print(f"开始测试模型: {model_name}")
    print(f"{'='*70}")
    
    tester = NetworkTester(model)
    
    train_loader, test_loader = get_default_mnist_loaders(batch_size)
    
    sample_data, _ = next(iter(test_loader))
    input_size = sample_data.shape
    
    print("\n[1/4] 模型摘要")
    tester.summary(input_size)
    
    print("\n[2/4] 数据流动可视化")
    save_path = f'/home/lbh/deep_learning_reproduction/test/{model_name}_data_flow.png'
    tester.visualize_data_flow(sample_data, save_path)
    
    print("\n[3/4] 训练测试")
    tester.test_training(train_loader, epochs=epochs)
    
    print("\n[4/4] 推理测试")
    accuracy = tester.test_inference(test_loader)
    
    tester.remove_hooks()
    
    print(f"\n{'='*70}")
    print(f"模型 {model_name} 测试完成！")
    print(f"推理准确率: {accuracy:.4f}")
    print(f"可视化图像: {save_path}")
    print(f"{'='*70}")
    
    return accuracy

if __name__ == '__main__':
    import sys
    sys.path.append('/home/lbh/deep_learning_reproduction/models')
    
    try:
        from day1_mlp import MLP
        model = MLP()
        run_full_test(model, model_name="MLP")
    except ImportError as e:
        print(f"导入模型失败: {e}")
        print("请确保模型文件在 models 目录下")