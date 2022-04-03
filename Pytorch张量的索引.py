#-------------------------------------------------------------
#              Pytorch 张量索引
#-------------------------------------------------------------

import torch

x = torch.Tensor([1,4,5,6,0,8,6,1,4,5])
print(x)
print(x[0]) # 第一个元素
print(x[9]) # 第十个元素
print(x[1:6]) # 输出从第2个元素到第6个元素

x = torch.randn((3,10))
print(x)
print(x[0]) # size 1x10
print(x[0,:]) # size 1x10
print(x[:,0]) # size 10x1
print(x[2,5:9])


x = torch.randn((4,10))
rows = [1,3]
columns = [2,9]
print(x[rows,columns]) #(2,3) (4,10)

x = torch.Tensor([1,4,5,6,0,8,6,1,4,5])
print(x[x>5])
print(x[(x>5) & (x<=6)]) # x>5 且 x<=6
print(x[(x>5) | (x<=6)]) # x>5 或 x<=6
print(x[(x>15)])

x = torch.Tensor([1,4,5,6,0,8,6,1,4,5])
print(torch.where(x>5, x, x/2)) #如果x>5, 则输出x, 否则输出x/2

x = torch.randn((4,10))
print(x.unique()) #输出至少出现过一次的元素
print(x.numel()) #输出x元素的个数