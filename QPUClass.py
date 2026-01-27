
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as mpatches
import copy
from Constants import Constants
import numpy as np
from scipy.stats import truncnorm

class QPU_base():
    # QPU的基类
    def __init__(self,id):
        self.id = id
        pass
    def get_conf(self):
        return self.nodes,self.edges
    def get_single_run_decoherence(self,num_qubits, mu_t1, cv_t1, mu_tphi, cv_tphi):
        """
        生成单次运行中 QPU 各个比特的 T1 和 T2 实现值
        """
        # 计算标准差 (sigma = mean * coefficient_of_variation)
        sigma_t1 = mu_t1 * cv_t1
        sigma_tphi = mu_tphi * cv_tphi

        def sample_truncated_normal(mean, std):
            # 在 [0, +inf] 范围内截断，确保物理意义正确 
            a, b = (0 - mean) / std, np.inf
            return truncnorm.rvs(a, b, loc=mean, scale=std, size=num_qubits)

        # 1. 为每个比特采样本次运行的 T1 [cite: 174]
        t1_list = sample_truncated_normal(mu_t1, sigma_t1)

        # 2. 为每个比特采样本次运行的 Tphi 
        tphi_list = sample_truncated_normal(mu_tphi, sigma_tphi)

        # 3. 计算对应的 T2 [cite: 135]
        # 公式: 1/T2 = 1/(2T1) + 1/Tphi
        t2_list = 1.0 / (1.0 / (2.0 * t1_list) + 1.0 / tphi_list)

        return t1_list, t2_list
    def update_off_set(self, offset):
        """
        将偏移量应用到节点列表和边列表上。
        
        Args:
            nodes (list): 原始节点列表，例如 [0, 1, 2, ...]
            edges (list): 原始边列表，例如 [(1, 2), (2, 3), ...]
            offset (int): 需要增加的偏移量
            
        Returns:
            tuple: (new_nodes, new_edges) 更新后的节点和边
        """
        # 1. 更新节点 ID
        # 每个节点 ID 加上 offset
        new_nodes = [node + offset for node in self.nodes]
        
        # 2. 更新边连接的节点 ID
        # 每一条边 (u, v) 变为 (u + offset, v + offset)
        new_edges = [(u + offset, v + offset) for u, v in self.edges]
        
        self.nodes = new_nodes
        self.edges = new_edges

    def get_dec_time(self):
        t1_vals, t2_vals = self.get_single_run_decoherence(
            self.numNodes, self.mu_t1, self.cv_t1, self.mu_tphi, self.cv_tphi
        )
        self.t1_Nodes = t1_vals
        self.t2_Nodes = t2_vals

        all_t1_values = list(self.t1_Nodes.values()) if isinstance(self.t1_Nodes, dict) else self.t1_Nodes
        all_t2_values = list(self.t2_Nodes.values()) if isinstance(self.t2_Nodes, dict) else self.t2_Nodes
    
        min_t1 = np.min(all_t1_values)
        min_t2 = np.min(all_t2_values)
    
        return min(min_t1, min_t2)# 该QPU的dec time


class QPU_Guadalupe(QPU_base):
    # Guadalupe 类
    def __init__(self,qid):
        super().__init__(qid)
        self.mu_t1 = 1700.0
        self.mu_tphi = 3400.0
        self.cv_t1 = 0.2
        self.cv_tphi = 0.2
        self.edges = [
            (1, 2), (2, 3), (3, 5), (5, 8),
            (8, 11), (11, 14), (14, 13), (13, 12),
            (12, 10), (10, 7), (7, 4), (4, 1),
            # Corners
            (6, 7), (0, 1), (8, 9), (12, 15)
        ]
        self.nodes = [i for i in range(16)]  # We have 16 nodes
        self.numNodes = len(self.nodes)
        self.numEdges = len(self.edges)
        self.dec_time = self.get_dec_time()

class QPU_PenguinV3(QPU_base):
    # V3 类
    def __init__(self,qid):
        super().__init__(qid)
        self.mu_t1 = 1700.0
        self.mu_tphi = 3400.0
        self.cv_t1 = 0.2
        self.cv_tphi = 0.2
        self.edges = [
            # 水平连接 (Horizontal edges)
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8), (8, 9),
            (10, 11), (11, 12), (12, 13), (13, 14),
            (15, 16), (16, 17), (17, 18), (18, 19),
            # 垂直连接 (Vertical edges)
            (0, 5), (5, 10), (10, 15),
            (1, 6), (6, 11), (11, 16),
            (2, 7), (7, 12), (12, 17),
            (3, 8), (8, 13), (13, 18),
            (4, 9), (9, 14), (14, 19)
        ]
        self.nodes = [i for i in range(20)]  # 20 个物理比特
        self.numNodes = len(self.nodes)
        self.numEdges = len(self.edges)
        self.dec_time = self.get_dec_time()

class QPU_PenguinV4(QPU_base):
    # v4类
    def __init__(self,qid):
        super().__init__(qid)
        self.mu_t1 = 1700.0
        self.mu_tphi = 3400.0
        self.cv_t1 = 0.2
        self.cv_tphi = 0.2
        self.edges = [
            (0, 1), (1, 2), (1, 4), (3, 4), (4, 7), 
            (5, 6), (6, 7), (7, 10), (8, 9), (9, 10), 
            (10, 12), (11, 12), (12, 15), (13, 14), (14, 15), 
            (15, 18), (16, 17), (17, 18), (18, 21), (19, 20), 
            (20, 21), (21, 24), (22, 23), (23, 24), (24, 25), (25, 26)
        ]
        self.nodes = [i for i in range(27)]  # 27 个物理比特
        self.numNodes = len(self.nodes)
        self.numEdges = len(self.edges)
        self.dec_time = self.get_dec_time()

class QPUConnection():
    ## NOTE: 加入异构退相干噪声 
    def __init__(self):
        self.pos = None #init - the positions of the nodes

        self.qpu_list = self.get_qpu_list()# QPU的列表，每个QPU包含点集合和边集合
        self.deadline = self.get_deadline()# 获取最小退相干时间

        self.G = self.create_DQC_graph()
        self.numNodes = len(self.G.nodes)
        self.numEdges = len(self.G.edges)

        # NOTE: 经典边和量子边的数量在这边修改
        n = len(Constants.QPU_Type)
        self.numEdgesQuantum = int(n*(n-1)/2)
        self.numEdgesClassic = len(self.G.edges) - self.numEdgesQuantum # WARNING: only for this specific DQC architecture

    def get_config_of_QPU(self,type,qid):
        # 获取不同的QPU型号的点集和边集
        if(type=="Guadalupe"):
            return QPU_Guadalupe(qid)
        elif(type=="PenguinV3"):
            return QPU_PenguinV3(qid)
        elif(type=="PenguinV4"):
            return QPU_PenguinV4(qid)

    def get_deadline(self):
        # 获取多个QPU的最小退相干时间
        deadline = -999
        for qpu in self.qpu_list:
            if qpu.dec_time > deadline:
                deadline = qpu.dec_time
        return deadline

    def get_qpu_list(self):
        qpu_list = []
        offset = 0
        qid = 0
        for qpu in Constants.QPU_Type:
            tmp = self.get_config_of_QPU(qpu,qid)
            tmp.update_off_set(offset)
            qpu_list.append(tmp)
            offset += len(tmp.nodes)
            qid += 1
        return qpu_list

    def create_DQC_graph(self):
        # 创建QPU集合图
        G = nx.Graph()
        sub_g_set = []
        for qty in self.qpu_list:
            # print(conf)
            g_tmp = self.create_single_graph(qty.get_conf())# conf is set of nodes and edges from different QPU
            sub_g_set.append(g_tmp)

        G = nx.disjoint_union_all(sub_g_set)

        import itertools

        connection_nodes = []
        current_offset = 0

        for i in range(len(sub_g_set)):
            num_nodes_current = sub_g_set[i].number_of_nodes()
            
            # 策略 A：如果您想用每个子图的“最后一个节点”作为连接点
            # node_index = current_offset + num_nodes_current - 1
            
            # 策略 B (更常见)：如果您想用每个子图的“第一个节点”作为连接点 (通常是 node 0)
            # 比如在 union 之后，第二个子图的 node 0 会变成 node 16
            node_index = current_offset 
            
            connection_nodes.append(node_index)
            
            # 更新偏移量，为下一个子图做准备
            current_offset += num_nodes_current

        # 2. 第二步：将收集到的连接点两两互联
        # itertools.combinations(list, 2) 会生成所有不重复的配对
        for u, v in itertools.combinations(connection_nodes, 2):
            # 在大图中添加边 (u, v)
            G.add_edge(u, v, label="quantum", tele_qubit=False)
            print("building quanutm edge","(",u,",",v,")")
            # 设置边的属性
            G.edges[(u, v)]['mask_generate'] = True
            G.edges[(u, v)]['mask_ghz_generate'] = True


        # 2. 遍历并建立quantum连接
        # current_offset = 0
        # for i in range(len(sub_g_set) - 1):
        #     # 当前子图的节点数量
        #     num_nodes_current = sub_g_set[i].number_of_nodes()
        #     # 下一个子图的节点数量
        #     num_nodes_next = sub_g_set[i+1].number_of_nodes()
    
        #     # 计算在大图 G 中的索引：
        #     # 当前子图的“末尾节点”索引 = 当前偏移量 + 节点数 - 1
        #     last_node_index = current_offset + num_nodes_current - 1
            
        #     # 下一个子图的“首个节点”索引 = 当前偏移量 + 当前子图节点数
        #     first_node_index = current_offset + num_nodes_current
            
        #     # 在大图中添加边
        #     G.add_edge(last_node_index, first_node_index, label="quantum", tele_qubit=False) # Add edge between node 0 of both graphs (In the union graph, node 0 of second graph is 16)
        #     G.edges[(last_node_index,first_node_index)]['mask_generate'] = True # 如果False则不能产生EPR
            
        #     # 更新偏移量，供下一轮循环使用
        #     current_offset += num_nodes_current

        # 验证结果

        # print("建立量子连接后的边数：",len(G.edges))
        # print(f"总节点数: {G.number_of_nodes()}")
        # print(f"新增的跨图连接: {[(u, v) for u, v in G.edges() if abs(u-v) == 1 and u < v]}")

        for node in G.nodes:
            G.nodes[node]['weight'] = 0  # Set your initial weight value in the nodes! (cooldown)


        nx.draw(G, with_labels=True, node_color='lightblue', node_size=300)
        plt.savefig("figs/DQC.png")
        return G

    def create_single_graph(self,conf):
        G = nx.Graph()
        G.add_nodes_from(conf[0])
        G.add_weighted_edges_from((u, v, 0) for u, v in conf[1])
        for edge in G.edges():
            G.edges[edge]['label'] = "simple"
            G.edges[edge]['mask_tele_qubit'] = True   #initialization of the masks, you can initially perform any action
            G.edges[edge]['mask_swap'] = True
            # G.edges[edge]['mask_tofoligate'] = False
        return G

    


    def Guadelupe_config(self):
        # return the edges and nodes configuration of IBM Q Guadalupe Quantum Processor
        edges = [
            (1, 2), (2, 3), (3, 5), (5, 8),
            (8, 11), (11, 14), (14, 13), (13, 12),
            (12, 10), (10, 7), (7, 4), (4, 1),
            # Corners
            (6, 7), (0, 1), (8, 9), (12, 15)
        ]
        nodes = [i for i in range(16)]  # We have 16 nodes
        return nodes, edges

    def Penguin_V4_config(self):
        # 返回 IBM Penguin V4 (Falcon r4/r5 27-qubit Heavy-hex) 处理器的配置
        # 该拓扑结构由重六角形单元组成
        edges = [
            (0, 1), (1, 2), (1, 4), (3, 4), (4, 7), 
            (5, 6), (6, 7), (7, 10), (8, 9), (9, 10), 
            (10, 12), (11, 12), (12, 15), (13, 14), (14, 15), 
            (15, 18), (16, 17), (17, 18), (18, 21), (19, 20), 
            (20, 21), (21, 24), (22, 23), (23, 24), (24, 25), (25, 26)
        ]
        nodes = [i for i in range(27)]  # 27 个物理比特
        return nodes, edges

    def Penguin_V3_config(self):
        # 返回 IBM Penguin V3 (20-qubit 4x5 Grid) 处理器的边和节点配置
        # 采用标准的网格连接方式
        edges = [
            # 水平连接 (Horizontal edges)
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8), (8, 9),
            (10, 11), (11, 12), (12, 13), (13, 14),
            (15, 16), (16, 17), (17, 18), (18, 19),
            # 垂直连接 (Vertical edges)
            (0, 5), (5, 10), (10, 15),
            (1, 6), (6, 11), (11, 16),
            (2, 7), (7, 12), (12, 17),
            (3, 8), (8, 13), (13, 18),
            (4, 9), (9, 14), (14, 19)
        ]
        nodes = [i for i in range(20)]  # 20 个物理比特
        return nodes, edges

q = QPUConnection()
q.create_DQC_graph()
    
    # # Function to create two Guadelupe-like graphs connected with a quantum link
    # ======= Original function implement ========= 
    # def create_DQC_graph(self):
    #     # Create 
    #     G1 = self.create_graph()
    #     G2 = self.create_graph()
    #     # Union the two graphs
    #     G = nx.disjoint_union(G1, G2)

    #     for node in G.nodes:
    #         G.nodes[node]['weight'] = 0  # Set your initial weight value in the nodes! (cooldown)

    #     # Add edge between node 0 of both graphs (In the union graph, node 0 of second graph is 16)
    #     G.add_edge(0, 16, weight=0, label="quantum", tele_qubit=False)# 一条Quantum Tunnel
    #     G.edges[(0,16)]['mask_generate'] = True # 如果False则不能产生EPR
    #     numNodes = len(G.nodes)
    #     numEdges = len(G.edges)

    #     # -----The code below just prints the coupling graph --------
    #     # Positions for nodes in a circular pattern
    #     circle_nodes = [1,2,3,5,8,11,14,13,12,10,7,4]
    #     angle = np.linspace(0, 2 * np.pi, len(circle_nodes) + 1)[:-1]
    #     pos1 = {node: (np.cos(a), np.sin(a)) for node, a in zip(circle_nodes, angle)}
    #     # Positions for the corner nodes
    #     corner_nodes = [0, 6, 9, 15]
    #     # Choose angles of corner nodes to be close to their nearest node
    #     corner_angles = [angle[circle_nodes.index(n)] for n in [1, 7, 8, 12]]
    #     # Position corner nodes closer to the circle (radius 1.5 instead of 2)
    #     pos1.update({node: (1.5 * np.cos(a), 1.5 * np.sin(a)) for node, a in zip(corner_nodes, corner_angles)})
    #     # The same positions for the second graph, shifted to the right and rotated 180 degrees
    #     rotation = np.pi
    #     shift = 4  # Increase shift from 3 to 4
    #     pos2 = {node + 16: (shift + np.cos(a + rotation), np.sin(a + rotation)) for node, a in zip(circle_nodes, angle)}
    #     pos2.update({node + 16: (shift + 1.5 * np.cos(a + rotation), 1.5 * np.sin(a + rotation)) for node, a in zip(corner_nodes, corner_angles)})
    #     # Combine the positions
    #     pos_G = {**pos1, **pos2}
    #     # Plot the graph
    #     fig, ax = plt.subplots()
    #     edges = G.edges()
    #     colors = ['red' if e in [(0, 16), (16, 0)] else 'black' for e in edges]
    #     nx.draw(G, pos_G, with_labels=True, ax=ax, edge_color=colors, node_color='lightblue', node_size=300)
    #     plt.show()
    #     plt.savefig("QPU.png")
    #     self.pos = pos_G
    #     return G

    # # Function to create a Guadelupe-like graph
    # def create_graph(self):
    #     G = nx.Graph()
    #     # Add nodes
    #     nodes = [i for i in range(16)]  # We have 16 nodes
    #     G.add_nodes_from(nodes)
        
    #     # Add edges 
    #     edges = [
    #         (1, 2), (2, 3), (3, 5), (5, 8),
    #         (8, 11), (11, 14), (14, 13), (13, 12),
    #         (12, 10), (10, 7), (7, 4), (4, 1),
    #         # Corners
    #         (6, 7), (0, 1), (8, 9), (12, 15)
    #     ]
    #     G.add_weighted_edges_from((u, v, 0) for u, v in edges)
    #     for edge in G.edges():
    #         G.edges[edge]['label'] = "simple"
    #         G.edges[edge]['mask_tele_qubit'] = True   #initialization of the masks, you can initially perform any action
    #         G.edges[edge]['mask_swap'] = True
    #     return G
    

    
















        
        









