import sys
sys.path.append('/home/lbh/deep_learning_reproduction/models')

import torch
import torch.nn as nn
from day3_Transformer import FullTransformer, train_loader, test_loader, device

def test_transformer():
    print("=" * 50)
    print("开始测试 Transformer 模型")
    print("=" * 50)
    
    # 初始化模型
    model = FullTransformer(
        img_size=32,
        patch_size=4,
        in_c=3,
        d_model=128,
        nhead=8,
        num_layers=6,
        dim_feedforward=256,
        num_classes=10
    ).to(device)
    
    print("\n1. 前向传播测试（查看数据形状变化）")
    print("-" * 50)
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        data, label = next(iter(train_loader))
        data, label = data.to(device), label.to(device)
        print(f"\n批次数据形状: {data.shape}")
        print("\n开始前向传播...")
        out = model(data)
        print(f"\n前向传播完成，输出形状: {out.shape}")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

if __name__ == '__main__':
    test_transformer()
