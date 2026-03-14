import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import scatter

# ==========================================
# 1. 物理层：用于 SNUH 的 HGNN 编码器
# ==========================================
class SNUH_HGNN_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SNUH_HGNN_Encoder, self).__init__()
        # 使用 PyG 自带的超图卷积层
        self.hgnn_conv1 = HypergraphConv(in_channels, hidden_channels)
        self.hgnn_conv2 = HypergraphConv(hidden_channels, out_channels)

    def forward(self, x_phy, hyperedge_index):
        """
        x_phy: 物理节点特征矩阵，维度 [|V|, C_phy]
        hyperedge_index: 超边索引矩阵，维度 [2, num_edges] (行0是节点ID，行1是超边ID)
        """
        # 第一层卷积
        z = self.hgnn_conv1(x_phy, hyperedge_index)
        z = F.relu(z)
        # 第二层卷积
        z = self.hgnn_conv2(z, hyperedge_index)
        
        # Readout: 全局平均池化 (将所有物理节点压缩为一个向量)
        # 对应论文中的 h_{H_t} = Readout(Z)
        h_H = torch.mean(z, dim=0) 
        
        return h_H

# ==========================================
# 2. 逻辑层：用于 TDAG 的 Structure2Vec 编码器
# ==========================================
class TDAG_Structure2Vec(nn.Module):
    def __init__(self, in_channels, hidden_channels, iterations=3):
        super(TDAG_Structure2Vec, self).__init__()
        self.iterations = iterations
        self.hidden_channels = hidden_channels
        
        # 对应论文中的可学习参数 theta1, theta2, theta3
        self.theta1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.theta2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.theta3 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        
    def forward(self, x_log, edge_index):
        """
        x_log: 逻辑门特征矩阵，维度 [|V_t|, C_log]
        edge_index: 有向图的边索引，维度 [2, num_edges] (前驱节点 -> 后继节点)
        """
        num_gates = x_log.size(0)
        
        # 初始化 mu_g^{(0)} 为 0 向量
        mu = torch.zeros((num_gates, self.hidden_channels), device=x_log.device)
        
        # 预先计算 theta1 * x_g^{log} (因为 x_log 在迭代中是不变的)
        x_embedded = self.theta1(x_log)
        
        src, dst = edge_index[0], edge_index[1]
        
        # T 次递归迭代消息传递
        for k in range(self.iterations):
            # 1. 聚合来自前驱节点 (Fan-in) 的信息: u \in N_{in}(g)
            # 即寻找目标节点是 g 的边，将其源节点 (src) 的 mu 累加到 dst
            msg_in = scatter(mu[src], dst, dim=0, dim_size=num_gates, reduce='sum')
            
            # 2. 聚合来自后继节点 (Fan-out) 的信息: u \in N_{out}(g)
            # 即寻找源节点是 g 的边，将其目标节点 (dst) 的 mu 累加到 src
            msg_out = scatter(mu[dst], src, dim=0, dim_size=num_gates, reduce='sum')
            
            # 3. 更新 mu_g^{(k+1)}，对应论文核心更新公式
            mu_next = x_embedded + self.theta2(msg_in) + self.theta3(msg_out)
            mu = F.relu(mu_next)
            
        # Readout: 全局求和池化 (Sum Pooling)
        # 对应论文中的 h_{G_t} = \sum \mu_g^{(T)}
        h_G = torch.sum(mu, dim=0)
        
        return h_G

# ==========================================
# 3. 主干网络：双图编码器 (DDQN 的输入层)
# ==========================================
class DQC_StateEncoder(nn.Module):
    def __init__(self, c_phy=4, c_log=5, h_dim_phy=64, h_dim_log=64, out_dim=128):
        super(DQC_StateEncoder, self).__init__()
        
        # 初始化 HGNN 和 S2V 模块
        self.hgnn = SNUH_HGNN_Encoder(in_channels=c_phy, hidden_channels=32, out_channels=h_dim_phy)
        self.s2v = TDAG_Structure2Vec(in_channels=c_log, hidden_channels=h_dim_log, iterations=3)
        
        # 降维层 (可选项)：将拼接后的高维向量映射到指定的 RL 状态维度
        self.fc_fusion = nn.Linear(h_dim_phy + h_dim_log, out_dim)
        
    def forward(self, snuh_data, tdag_data):
        """
        接收环境传来的双图数据，返回固定长度的 Embedding Vector。
        """
        # 1. 编码物理硬件图 (SNUH)
        x_phy = snuh_data['x']                # [num_phy_nodes, C_phy]
        hyperedges = snuh_data['edge_index']  # [2, num_hyperedges]
        h_H = self.hgnn(x_phy, hyperedges)    # 输出维度: [h_dim_phy]
        
        # 2. 编码量子逻辑图 (TDAG)
        x_log = tdag_data['x']                # [num_gates, C_log]
        edges = tdag_data['edge_index']       # [2, num_directed_edges]
        h_G = self.s2v(x_log, edges)          # 输出维度: [h_dim_log]
        
        # 3. 拼接两个全局特征
        # 对应论文中的 h_t = [h_{H_t} || h_{G_t}]
        h_t = torch.cat([h_H, h_G], dim=-1)   # 输出维度: [h_dim_phy + h_dim_log]
        
        # (可选) 融合为一个最终维度的 Embedding Vector
        state_embedding = F.relu(self.fc_fusion(h_t))
        
        return state_embedding