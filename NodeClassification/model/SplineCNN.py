import torch.nn.functional as F
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SplineConv

class SplineCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SplineCNN, self).__init__()
        # 定义第一层SplineConv，输入特征维度为num_features，输出特征维度为16，伪坐标维度为2，卷积核大小为5
        # dim参数表示伪坐标维度，即边特征的维度，默认为1，可以增加边的信息量和表达能力
        # kernel_size参数表示卷积核大小，默认为2，可以控制卷积操作的局部性和感受野范围
        self.conv1 = SplineConv(in_channels, 16, dim=2, kernel_size=5)
        # 定义第二层SplineConv，输入特征维度为16，输出特征维度为num_classes，伪坐标维度为2，卷积核大小为5
        self.conv2 = SplineConv(16, out_channels, dim=2, kernel_size=5)

    def forward(self, data : Data):
        # 获取节点特征矩阵x、邻接矩阵edge_index和边特征矩阵edge_attr
        x = data.x
        edge_index = data.edge_index

        # 第一层SplineConv的前向传播，使用ReLU激活函数，并进行dropout操作（丢弃率为0.5）
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.nn.functional.dropout(x, p=0.5)

        # 第二层SplineConv的前向传播，不使用激活函数，并进行dropout操作（丢弃率为0.5）
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.dropout(x, p=0.5)

        # 返回最终的节点特征矩阵x，每一行表示一个节点的类别预测向量
        return x