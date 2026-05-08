import sys
sys.path.append('/home/lbh/deep_learning_reproduction/models')
sys.path.append('/home/lbh/deep_learning_reproduction/test')

from day1_mlp import MLP
from test_interface import NetworkTester, get_default_mnist_loaders

model = MLP()
tester = NetworkTester(model)

train_loader, test_loader = get_default_mnist_loaders(batch_size=32)

sample_data, _ = next(iter(test_loader))
input_size = sample_data.shape

print("[1/4] 模型摘要")
tester.summary(input_size)

print("\n[2/4] 数据流动可视化")
tester.visualize_data_flow(sample_data, save_path='/home/lbh/deep_learning_reproduction/test/simple_test.png')

print("\n[3/4] 训练测试 (1轮)")
tester.test_training(train_loader, epochs=1)

print("\n[4/4] 推理测试")
accuracy = tester.test_inference(test_loader)

tester.remove_hooks()

print(f"\n测试完成！准确率: {accuracy:.4f}")