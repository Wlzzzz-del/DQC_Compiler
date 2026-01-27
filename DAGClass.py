
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as mpatches
import copy
from Constants import Constants





class DAGClass():
    
    def __init__(self):
        # Create a Directed Graph
        if Constants.IF_ENABLE_TOFFOLIOS:
            # self.DAG = self.create_toffoli_DAG(Constants.NUMQ, Constants.NUMG)# number of Qubits and Number of Gates
            self.DAG = self.create_mixed_DAG(Constants.NUMQ,Constants.NUMG)
        else:
            self.DAG = self.create_random_DAG(Constants.NUMQ, Constants.NUMG)# number of Qubits and Number of Gates

        self.tofoli_gate_prob = 0.3
        #self.DAG = self.create_DAG()
        self.topo_order = self.compute_topo_order()
        self.numGates = len(self.topo_order)  #initial number of gates
        self.layers = self.compute_node_layers()
        #self.numQubits = self.compute_numQubits()
        self.numQubits = Constants.NUMQ
        
    def generate_cnot_dag(self, numQ, numG):
        # 这个函数的作用是随机生成CNOT gate，并为每个门分配一个时间层级（Layer）
        # 由于量子比特在同一时间只能执行一个操作，该门必须被放置在 max(layer_x, layer_y) + 1 的位置。
        qubit_layers = {qubit: 0 for qubit in range(numQ)}
        dag = []
        for _ in range(numG):
            x, y = random.sample(range(numQ), 2)# 随机采样两个Node
            least_layer = max(qubit_layers[x], qubit_layers[y]) + 1# 计算这两个门的当前最后层数
            qubit_layers[x], qubit_layers[y] = least_layer, least_layer # 赋值最后层数
            dag.append((x, y, least_layer-1))# 放入dag中
        dag.sort(key=lambda node: node[2])# 按照时间层级排序
        return dag

    def create_random_DAG(self,numQ, numG):
        dag_list = self.generate_cnot_dag(numQ, numG)
        DAG = nx.DiGraph()
        # Keeps track of the most recent node for each qubit
        qubit_most_recent_node = {}
        
        for x, y, l in dag_list:  # remember it is sorted
            current_node = (x, y, l)
            DAG.add_node(current_node)# 往DAG添加edge
            
            # Connect to the most recent node for each qubit, if it exists
            # 如果量子比特 $x$ 刚执行完门 $A$，现在要执行门 $B$，则会连接一条从 $A$ 到 $B$ 的有向边。
            for qubit in [x, y]:
                if qubit in qubit_most_recent_node:
                    prev_node = qubit_most_recent_node[qubit]
                    DAG.add_edge(prev_node, current_node)
            
            # Update the most recent node for the involved qubits
            qubit_most_recent_node[x] = current_node
            qubit_most_recent_node[y] = current_node
        
        return DAG


    def create_DAG(self):
        DAG = nx.DiGraph()
        DAG.add_edges_from([
            ((0,1,0), (0,2,1)), ((0,2,1), (0,4,2)), ((0,2,1), (2,3,2)), ((0,4,2), (0,2,3)), ((2,3,2), (0,2,3)), ((0,2,3), (0,1,4)),
            ((0,1,0), (1,5,1)), ((1,5,1), (1,6,2)), ((1,6,2), (0,1,4)),  
        ])
        # DAG.add_edges_from([
        #     ((0,1,0), (0,2,1))
        # ])
        return DAG
    
    def compute_topo_order(self):
        # Get nodes in topological order
        topo_order = list(nx.topological_sort(self.DAG))
        print("topo_order is: ", topo_order)
        # Function to compute layer of each node for better visualization
        # Compute layers of nodes
        return topo_order
            
    def compute_node_layers(self):
        layers = {node: 0 for node in self.topo_order}
        for node in self.topo_order:
            for pred in self.DAG.predecessors(node):
                layers[node] = max(layers[node], layers[pred] + 1)
        return layers
    

        

    # def remove_node(self, node): #It does not check whether it is possible to implement the gate
    #     ball1, ball2 = node
    #     # Find nodes with matching first and second elements.
    #     matching_nodes = [node for node in self.DAG if (node[0] == ball1 and node[1] == ball2) or (node[1] == ball1 and node[0] == ball2)]
    #     # Return the node with the smallest third element. We need it to remove the correct gate (the first that appears in the layering)
    #     node_to_remove = min(matching_nodes, key=lambda node: node[2]) if matching_nodes else None
    #     # Remove the node from the graph.
    #     self.DAG.remove_node(node_to_remove)
    #     # Remove the node from topo_order.
    #     self.topo_order.remove(node_to_remove)
    #     print("DAG nodes after removal", self.DAG.nodes)


    # def print_DAG(self):
    #     # Create a dictionary of positions based on topological order and layer
    #     pos = {node: (i%3, self.layers[node]) for i, node in enumerate(self.topo_order)}
    #     # Draw the Directed Graph
    #     fig, ax = plt.subplots()
    #     nx.draw(self.DAG, pos, with_labels=True, node_color='lightblue', node_size=1500, ax=ax)
    #     plt.savefig("figs/tofo_DAG.png")
    #     plt.show()
    #     return

    def compute_numQubits(self):
        # Use a set to store unique numbers from the first two components of each node
        unique_numbers = set()
        # Iterate through all nodes in the DAG
        for node in self.DAG.nodes:
            # Add the first component of the node to the set
            unique_numbers.add(node[0])
            # Add the second component of the node to the set
            unique_numbers.add(node[1])
        # The number of unique qubits is the size of the set
        numQubit = len(unique_numbers)
        return numQubit

# ========= 新增的函数 添加toffoli dag=========== 
    def generate_toffoli_dag(self, numQ, numG):
        # qubit_layers 追踪每个比特当前所在的最后层数
        qubit_layers = {qubit: 0 for qubit in range(numQ)}
        dag = []
        
        for _ in range(numG):
            # 1. 随机采样 3 个不重复的量子比特：两个控制位，一个目标位
            c1, c2, t = random.sample(range(numQ), 3)
            
            # 2. 计算这三个比特中最大的层数，确保逻辑时序正确
            least_layer = max(qubit_layers[c1], qubit_layers[c2], qubit_layers[t]) + 1
            
            # 3. 更新这三个比特的层数状态
            qubit_layers[c1] = qubit_layers[c2] = qubit_layers[t] = least_layer
            
            # 4. 存入 dag 列表 (c1, c2, t, layer_index)
            dag.append((c1, c2, t, least_layer - 1))
        
        # 按层级排序
        dag.sort(key=lambda node: node[3])
        return dag

    def create_toffoli_DAG(self, numQ, numG):
        dag_list = self.generate_toffoli_dag(numQ, numG)
        # print(dag_list)
        DAG = nx.DiGraph()
        qubit_most_recent_node = {}
        
        for c1, c2, t, l in dag_list:
            current_node = (c1, c2, t, l)
            DAG.add_node(current_node)
            
            # 为该门涉及的三个量子比特分别建立依赖边
            for qubit in [c1, c2, t]:
                if qubit in qubit_most_recent_node:
                    prev_node = qubit_most_recent_node[qubit]
                    DAG.add_edge(prev_node, current_node)
            
            # 更新这三个比特最近一次参与的节点记录
            qubit_most_recent_node[c1] = current_node
            qubit_most_recent_node[c2] = current_node
            qubit_most_recent_node[t] = current_node
            
        return DAG

    def print_DAG(self):
            """
            绘制 DAG 图。
            布局策略：
            X轴: 时间层级 (Layer)
            Y轴: 涉及比特的平均位置 (Qubit Index Average)
            """
            plt.figure(figsize=(12, 6)) # 设置画布大小，宽一点更清晰
            
            pos = {}
            for node in self.DAG.nodes:
                # node 结构示例: (c1, c2, t, layer) 或 (c, t, layer)
                # 最后一个元素是 layer，前面的是 qubit indices
                qubits = node[:-1]
                layer = node[-1]
                
                # X 坐标: 时间层级
                x = layer
                
                # Y 坐标: 为了防止重叠，使用比特索引的平均值
                # 例如 (0, 1, layer) -> y = 0.5
                # 例如 (2, 4, layer) -> y = 3.0
                y = sum(qubits) / len(qubits)
                
                # 这里的 -y 是为了让 Qubit 0 在顶部，符合通常的电路图习惯
                pos[node] = (x, -y) 

            # 绘制图形
            # node_size 可以根据标签长度动态调整，或者设个固定值
            nx.draw(self.DAG, pos, 
                    with_labels=True, 
                    node_color='lightblue', 
                    edge_color='gray',
                    node_size=2000, 
                    font_size=8,
                    arrowsize=20)
            
            plt.title("Quantum Circuit DAG Visualization")
            plt.xlabel("Layer (Time Step)")
            plt.ylabel("Qubit Index (Approx)")
            
            # 移除坐标轴刻度让图更干净
            plt.axis('on') # 或者 'off'
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.savefig("figs/tofo_DAG.png")
            plt.show()
            # return

    def remove_node(self, node): 
            """
            移除指定的节点。
            逻辑：查找所有涉及相同量子比特集合的门，并移除其中层级（Layer）最早的一个。
            兼容 CNOT (2-qubit) 和 Toffoli (3-qubit)。
            """
            # 获取目标节点涉及的比特集合（排除最后一个元素 layer）
            target_qubits = set(node[:-1]) 
            
            matching_nodes = []
            for n in self.DAG.nodes:
                # 获取当前遍历节点涉及的比特集合
                current_qubits = set(n[:-1])
                
                # 比较比特集合是否相同 (忽略顺序，例如 (0,1) 和 (1,0) 视为相同)
                if current_qubits == target_qubits:
                    matching_nodes.append(n)
            
            # 找到层级最早的节点（元组的最后一个元素是 layer）
            if matching_nodes:
                node_to_remove = min(matching_nodes, key=lambda x: x[-1])
                
                # 从图中移除
                self.DAG.remove_node(node_to_remove)
                
                # 从拓扑序列表中移除 (如果存在)
                if node_to_remove in self.topo_order:
                    self.topo_order.remove(node_to_remove)
                
                print(f"Removed node: {node_to_remove}")
                print("DAG nodes count after removal:", self.DAG.number_of_nodes())
            else:
                print("Node not found or already removed.")

    def generate_mixed_dag_list(self, numQ, numG, toffoli_prob=0.3):
        """
        生成混合了 CNOT 和 Toffoli 的门列表。
        toffoli_prob: 生成 Toffoli 门的概率 (0.0 - 1.0)
        """
        qubit_layers = {qubit: 0 for qubit in range(numQ)}
        dag_list = []
        
        for _ in range(numG):
            # 1. 决定生成哪种门
            is_toffoli = random.random() < toffoli_prob
            
            if is_toffoli:
                # --- 生成 Toffoli (3 qubits) ---
                # 采样 3 个不同比特
                if numQ < 3: raise ValueError("Need at least 3 qubits for Toffoli")
                c1, c2, t = random.sample(range(numQ), 3)
                
                # 计算层级 (取 3 个比特的最大值)
                least_layer = max(qubit_layers[c1], qubit_layers[c2], qubit_layers[t]) + 1
                
                # 更新状态
                qubit_layers[c1] = qubit_layers[c2] = qubit_layers[t] = least_layer
                
                # 存入节点：建议带上类型标签，或者直接存元组
                # # NOTE: Toffoli格式: ('Toffoli', c1, c2, t, layer)
                dag_list.append(('Toffoli', c1, c2, t, least_layer - 1))
                
            else:
                # --- 生成 CNOT (2 qubits) ---
                # 采样 2 个不同比特
                c, t = random.sample(range(numQ), 2)
                
                # 计算层级 (取 2 个比特的最大值)
                least_layer = max(qubit_layers[c], qubit_layers[t]) + 1
                
                # 更新状态
                qubit_layers[c] = qubit_layers[t] = least_layer
                
                # 存入节点
                # NOTE: CNOT格式: ('CNOT', c, t, layer)
                dag_list.append(('CNOT', c, t, least_layer - 1))
        
        # 按时间层级排序
        dag_list.sort(key=lambda node: node[-1])
        return dag_list

    def create_mixed_DAG(self, numQ, numG, toffoli_prob=0.3):
        dag_list = self.generate_mixed_dag_list(numQ, numG, toffoli_prob)
        DAG = nx.DiGraph()
        qubit_most_recent_node = {}
        
        for gate_info in dag_list:
            # 1. 解析节点信息
            gate_type = gate_info[0]
            layer = gate_info[-1]

            # NOTE: 后面节点的操作都得按这个逻辑来 
            if gate_type == 'Toffoli':
                # 解包: Type, c1, c2, t, layer
                _, c1, c2, t, _ = gate_info
                involved_qubits = [c1, c2, t]
                # 节点存储：可以存完整元组，也可以存简化的
                current_node = (c1, c2, t, layer) # 这里存为 4元组
            else:
                # 解包: Type, c, t, layer
                _, c, t, _ = gate_info
                involved_qubits = [c, t]
                # 节点存储：存为 3元组，以示区分，或者补齐为 (c, t, -1, layer)
                current_node = (c, t, layer) 
            
            # 添加节点
            # 注意：为了让后续代码（如 print_DAG）能区分类型，建议节点中包含类型
            # 或者在这里统一格式。为了演示，我这里保留了混合长度元组。
            DAG.add_node(current_node)
            
            # 2. 建立依赖边 (逻辑对所有门通用)
            for qubit in involved_qubits:
                if qubit in qubit_most_recent_node:
                    prev_node = qubit_most_recent_node[qubit]
                    DAG.add_edge(prev_node, current_node)
            
            # 3. 更新最近节点记录
            for qubit in involved_qubits:
                qubit_most_recent_node[qubit] = current_node
                
        return DAG

# try_ = DAGClass()
# try_.print_DAG()