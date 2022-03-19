#-------------------------------------------------------------
#              Pytorch 张量Broadcast
#-------------------------------------------------------------

# 导入pytorch
import torch

# 基本数学运算
x = torch.tensor([2,2,2], dtype = torch.float32)
y = torch.tensor([3,4,5], dtype = torch.float32)

# 加法
out = torch.add(x,y)
print(out)

out = x + y
print(out)

# 减法
out = x - y
print(out)

out = torch.sub(x,y)
print(out)

# 除法
out = torch.div(x,y)
print(out)

out = x / y
print(out)

# 乘法
out = torch.mul(x,y)
print(out)

out = x * y
print(out)

# 矩阵乘法
x = torch.rand((2,5))
y = torch.rand((5,3))

out = torch.mm(x,y) # 2*3
print(out)
print(out.shape)

# 批量矩阵乘法
# 批量batch
batch = 16
c1 = 5
c2 = 10
c3 = 20

x1 = torch.rand(batch,c1,c2) #16*5*10
x2 = torch.rand(batch,c2,c3) #16*10*20

out = torch.bmm(x1,x2) #16*5*20
print(out.shape)

#指数
x = torch.tensor([[2,2,2],[2,2,2]])
out = x.pow(3)
print(out)

out = x ** 3
print(out)

#矩阵指数
x = torch.tensor([[1,1],[1,1]]) # 2*2
out = x.matrix_power(2)
print(out) #2*2

out = x.pow(2)
print(out)

# broadcasting
x1 = torch.rand((1,3))
x2 = torch.rand((3,3))

out = x1 - x2
print(out)

x1 = torch.rand((2,2))
x2 = torch.randint(1,10,(2,2))
out = x1.pow(x2)
print(out)

x2 = torch.randint(1,10,(1,2))
out = x1.pow(x2)
print(out)
