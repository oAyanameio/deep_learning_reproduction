import torch
from models.model import SimpleModel

# 简单的训练示例
model = SimpleModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 生成示例数据
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# 训练循环
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
