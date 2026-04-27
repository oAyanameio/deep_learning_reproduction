# Deep Learning Reproduction

[![GitHub stars](https://img.shields.io/github/stars/oAyanameio/deep_learning_reproduction.svg?style=social&label=Stars)](https://github.com/oAyanameio/deep_learning_reproduction)
[![GitHub forks](https://img.shields.io/github/forks/oAyanameio/deep_learning_reproduction.svg?style=social&label=Forks)](https://github.com/oAyanameio/deep_learning_reproduction)

## 项目介绍

这是一个深度学习模型复现项目，包含了多种经典深度学习模型的实现，主要用于学习和研究深度学习算法。项目使用PyTorch框架，实现了从简单的多层感知机（MLP）到复杂的卷积神经网络（CNN）等多种模型。

## 项目结构

```
deep_learning_reproduction/
├── config/          # 配置文件
├── data/            # 数据集（自动下载）
├── models/          # 模型定义
│   ├── day1_mlp.py  # MLP模型（MNIST分类）
│   └── day2_CNN.py  # CNN模型（CIFAR-10分类）
├── utils/           # 工具函数
├── train.py         # 训练脚本
├── requirements.txt # 依赖包
└── README.md        # 项目说明
```

## 依赖要求

项目使用以下依赖包：

- torch
- torchvision
- torchaudio
- tensorflow
- numpy
- pandas
- matplotlib
- scikit-learn
- jupyter

## 安装步骤

1. 克隆项目到本地：

```bash
git clone git@github.com:oAyanameio/deep_learning_reproduction.git
cd deep_learning_reproduction
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 项目计划

| 日期       | 任务                   | 状态       | 备注                     |
|------------|------------------------|------------|--------------------------|
| 2026-04-27 | 初始化项目结构         | 已完成     | 创建基本目录和文件结构   |
| 2026-04-27 | 实现MLP模型（MNIST）   | 已完成     | 包含LinearModel和MLP     |
| 2026-04-27 | 实现CNN模型（CIFAR-10） | 已完成     | 包含LeNet5、AlexNet、VGG16 |
| 2026-04-27 | 编写README文件         | 已完成     | 包含项目介绍和使用说明   |
| 2026-04-28 | 优化模型性能           | 待完成     | 调整超参数，提升准确率   |
| 2026-04-29 | 添加新模型             | 待完成     | 实现ResNet等高级模型     |
| 2026-04-30 | 编写单元测试           | 待完成     | 确保代码质量和稳定性     |
| 2026-05-01 | 文档完善               | 待完成     | 添加详细的API文档        |

## 模型介绍

### 1. 多层感知机 (MLP)

实现于 `models/day1_mlp.py`，用于MNIST手写数字分类：

- **LinearModel**：简单的线性模型
- **MLP**：多层感知机模型，包含两个隐藏层

### 2. 卷积神经网络 (CNN)

实现于 `models/day2_CNN.py`，用于CIFAR-10图像分类：

- **LeNet5**：经典的LeNet-5模型
- **AlexNet**：AlexNet模型（适配CIFAR-10）
- **VGG16**：VGG16模型（适配CIFAR-10）

## 使用方法

### 运行MLP模型

```bash
python models/day1_mlp.py
```

### 运行CNN模型

```bash
python models/day2_CNN.py
```

### 自定义训练

可以修改 `train.py` 文件来自定义训练过程。

## 数据集

项目自动下载以下数据集：

- **MNIST**：手写数字数据集，用于MLP模型
- **CIFAR-10**：彩色图像分类数据集，用于CNN模型

数据集会自动下载到 `data/` 目录，该目录已被添加到 `.gitignore` 中，不会被提交到GitHub。

## 训练结果

训练过程中会生成训练曲线并保存为 `training_curves.png`，包含训练损失和测试准确率的变化。

## 项目特点

- 代码结构清晰，易于理解和扩展
- 支持GPU加速（自动检测并使用GPU）
- 包含完整的训练和评估流程
- 提供详细的模型实现和注释

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目采用MIT许可证。

## 联系方式

- GitHub: [oAyanameio](https://github.com/oAyanameio)

---

*Note: This project is for educational purposes only.*