#-------------------------------------------------------------
#              Pytorch 张量初始化
#-------------------------------------------------------------

# 导入 pytorch 库
import torch

# 初始化张量
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor([[1,2,3],[4,5,6]],dtype = torch.float32, device = DEVICE, requires_grad = True)

print(x)
print(x.dtype)
print(x.device)
print(x.requires_grad)

# 其它初始化方法
x = torch.rand((2,3)) # rand 均匀分布
print(x)

x = torch.randn((2,3)) # randn 正态分布
print(x)

x = torch.randint(3,10,(2,3)) # randint 均匀分布取整数
print(x)

input = torch.rand((3,3))
x = torch.rand_like(input) # rand_like 用rand来取与input同样维度的矩阵
print(x)

x = torch.zeros((2,3)) # 生成0矩阵
print(x)

x = torch.ones((2,3)) # 生成1矩阵
print(x)

x = torch.eye(3,3) # 生成对角单位1矩阵
print(x)

x = torch.arange(start=0, end=10, step=1) #以step为间隔输出从start到end的张量列表， 注意不包含end值
print(x)

x = torch.linspace(start=0, end=9, steps=11) #返回一个1维张量，包含在区间start和end上均匀间隔的steps的值
print(x)

x = torch.diag(torch.rand(5)) #生成对角矩阵
print(x)

#-------------------------------------------------------------
# Numpy 与 Pytorch 数据类型转换
#-------------------------------------------------------------

# 导入Numpy包
import numpy as np

# 定义一个Ndarray对象
x = np.zeros((2,3))
print(x)

# 从 Numpy的Array 到 Torch的Tensor 的转换
x_torch = torch.from_numpy(x)
print(x_torch)

# 从 Torch的Tensor 到 Numpy的Array 的转换
x_back = x_torch.numpy()
print(x_back)