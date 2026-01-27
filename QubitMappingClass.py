import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as mpatches
import copy
from Constants import Constants

class QubitMappingClass():
    
    def __init__(self, qpu_list, dag,numNodes, numQubits, numEPR_threshold,numGHZ_threshold, initial_mapping=None):
        self.numNodes = numNodes
        self.numQubits = numQubits
        self.numEPR_threshold = numEPR_threshold
        self.numGHZ_threshold = numGHZ_threshold

        self.qpu_list = qpu_list# QPU的列表，包括每个QPU的node set和edge set
        self.dag = dag# 电路图

        # 对于不同指标的权重
        self.weights = {'W_T': 10, 'W_C': 1, 'lambda': 1.0}

        # 建立Node->QPU的映射
        self.node_to_qpu_id = {}
        for qpu in self.qpu_list:
            nodes = qpu.nodes
            q_id = qpu.id
            for node in nodes:
                self.node_to_qpu_id[node] = q_id

        print("[DEBUGing] node to qpu id",self.node_to_qpu_id)

        # 存放EPR的两个qubit吗？
        # EPRID->[box1,box2]
        self.ball_to_box = {}  # ball to box mapping
        # box1->EPRID, box2->EPRID
        self.box_to_ball = {}  # box to ball mapping
        # EPRID->[box1,box2]
        self.EPR_pairs = {}  # EPR pairs mapping
        self.EPR_pool = [f"EPR-{i}" for i in range(numEPR_threshold)]  # Pool of EPR IDs

        # === 新增：GHZ 态相关的存储 ===
        # GHZ_ID -> [box1, box2, box3]
        self.GHZ_ball_to_box = {}
        self.GHZ_box_to_ball = {}
        self.GHZ_triplets = {} 
        # GHZ ID 池 (类似于 EPR pool)
        self.GHZ_pool = [f"GHZ-{i}" for i in range(numGHZ_threshold)]

        if initial_mapping is None:
            initial_mapping = self.generate_random_initial_mapping()
        print("DEBUG: RUNNING DSF MAPPING!")
        if Constants.USE_DSF_MAPPING:
            initial_mapping = self.DSF_mapping()
        else:
            initial_mapping = self.generate_random_initial_mapping()


        # Initialize with given mapping
        if initial_mapping is not None:
            for ball, box in initial_mapping.items():
                if (ball > numQubits-1 or box > numNodes - 1):
                    raise Exception("Ball or box out of limit.")
                self.ball_to_box[ball] = box
                self.box_to_ball[box] = ball
                self.GHZ_ball_to_box[ball] = box
                self.GHZ_box_to_ball[box] = ball
        else:   
            raise Exception("Error - initial mapping is None")

    def get_box(self, ball):
        # 获得ball 对应的 box
        if ball not in self.ball_to_box:
            raise Exception(f"No box found for ball {ball}.")
        return self.ball_to_box[ball]

    def get_ball(self, box):
        # 获得 box 对应的 ball
        if box not in self.box_to_ball:
            return self.box_to_ball.get(box, None)
        return self.box_to_ball[box]
    
    def get_ghz_ball(self, box):
        if box not in self.GHZ_box_to_ball:
            return self.GHZ_box_to_ball.get(box, None)
        return self.GHZ_box_to_ball[box]

    def generate_EPR_pair(self, box1, box2):
        # 产生EPR的函数
        # 先验证是否符合条件然后在各个list dict中记录
        if len(self.EPR_pool) == 0:
            raise Exception("No more EPR IDs available in the pool.")
        
        if (box1 > self.numNodes - 1 or box2 > self.numNodes - 1):
            raise Exception("Ball or box out of limit.")
        
        epr_id = self.EPR_pool.pop(0)  # Get the first available ID and remove it from the pool

        # Update the mappings
        self.ball_to_box[epr_id] = [box1, box2]
        self.box_to_ball[box1] = epr_id
        self.box_to_ball[box2] = epr_id
        self.EPR_pairs[epr_id] = [box1, box2]

    def destroy_EPR_pair(self, epr_id):
        # 销毁EPR
        # 先验证是否符合条件然后在各个list dict中删除
        if epr_id not in self.EPR_pairs:
            raise Exception(f"No EPR pair with ID {epr_id} exists.")

        # Remove the pair from the mappings
        self.EPR_pairs.pop(epr_id)
        boxes = self.ball_to_box[epr_id]  #might have been updated boxes
        self.ball_to_box.pop(epr_id)
        self.box_to_ball.pop(boxes[0])
        self.box_to_ball.pop(boxes[1])
        # Return the ID to the EPR pool
        self.EPR_pool.append(epr_id)
        self.EPR_pool.sort()  # Keep the pool sorted for predictability

    def query_GHZ_pair(self, ghz_id):
        if ghz_id not in self.GHZ_triplets:
            raise Exception(f"No EPR pair with ID {ghz_id} exists.")
        boxes = self.GHZ_ball_to_box[ghz_id]
        self.GHZ_triplets[ghz_id] = boxes
        return boxes

    def query_EPR_pair(self, epr_id):
        if epr_id not in self.EPR_pairs:
            raise Exception(f"No EPR pair with ID {epr_id} exists.")
        # Return the boxes associated with the EPR pair
        # Fetch the current boxes associated with the EPR pair from ball_to_box mapping
        boxes = self.ball_to_box[epr_id]
        # Update the EPR_pairs mapping
        self.EPR_pairs[epr_id] = boxes
        # Return the updated boxes
        return boxes
    
    def generate_random_initial_mapping(self):
        # 随机mapping
        if self.numQubits > self.numNodes:
            raise ValueError("Number of logical qubits cannot be greater than the number of physical qubits.")
        physical_qubits = list(range(self.numNodes))
        random.shuffle(physical_qubits)
        initial_mapping = {i: physical_qubits[i] for i in range(self.numQubits)}
        return initial_mapping

    ### ========= 新增的部分 ===========

    def _calculate_comm_cost(self, mapping):
        """计算通信代价：遍历 DAG 中的所有门"""
        comm_cost = 0
        W_T = self.weights['W_T']
        W_C = self.weights['W_C']
        
        # self.dag.DAG.nodes 存储了所有的门操作
        # 节点格式可能是 (c, t, layer) 或 (c1, c2, t, layer)
        for node in self.dag.DAG.nodes:
            # 获取该门涉及的所有逻辑比特 (去掉最后一个 layer 元素)
            involved_logic_qubits = node[:-1]
            
            # 查找这些逻辑比特所在的 QPU
            involved_qpu_ids = set()
            valid_gate = True
            
            for q in involved_logic_qubits:
                if q not in mapping:
                    # 如果映射中没有这个逻辑比特（可能是EPR或其他），暂时跳过或报错
                    # 这里假设 initial_mapping 覆盖了所有用于计算的逻辑比特
                    valid_gate = False 
                    break
                
                phys_node = mapping[q]
                if phys_node in self.node_to_qpu_id:
                    involved_qpu_ids.add(self.node_to_qpu_id[phys_node])
                else:
                    # 物理节点未分配 QPU (异常情况)
                    pass 

            if not valid_gate: continue

            # 如果涉及的 QPU 数量 > 1，说明跨芯片
            if len(involved_qpu_ids) > 1:
                # 判断门类型：根据涉及比特数判断
                # 3个比特 -> Toffoli (4元组减去layer)
                if len(involved_logic_qubits) == 3:
                    comm_cost += W_T
                else:
                    comm_cost += W_C
                    
        return comm_cost

    def _calculate_dsf_cost(self, mapping):
        """计算 DSF 代价：负载密度 / 退相干时间"""
        # 初始化 QPU 负载计数
        qpu_loads = {qpu.id: 0 for qpu in self.qpu_list}
        
        # 统计分布
        for log_q, phys_n in mapping.items():
            if phys_n in self.node_to_qpu_id:
                q_id = self.node_to_qpu_id[phys_n]
                qpu_loads[q_id] += 1
        
        dsf_total = 0
        for qpu in self.qpu_list:
            q_id = qpu.id
            capacity = qpu.numNodes # 若未指定容量，默认为节点数
            t_coh = qpu.dec_time # 防止除零
            
            # Density = Load / Capacity
            density = qpu_loads[q_id] / capacity
            
            # DSF Term
            dsf_total += density / t_coh
            
        return dsf_total

    def _get_total_cost(self, mapping):
        c_comm = self._calculate_comm_cost(mapping)
        c_dsf = self._calculate_dsf_cost(mapping)
        return c_comm + self.weights['lambda'] * c_dsf

    def DSF_mapping(self, max_iter=200, patience=50):
        """
        执行 DSF-TAP 算法搜索最优初始映射。
        搜索结束后，会自动调用 self._apply_mapping 更新类状态。
        """
        # 1. 从当前状态（可能是随机的）开始
        # 提取当前的逻辑->物理映射
        current_mapping = copy.deepcopy(self.ball_to_box)
        # 过滤掉 EPR 对 (key 是 str 'EPR-x')，只保留逻辑比特 (key 是 int)
        current_mapping = {k: v for k, v in current_mapping.items() if isinstance(k, int)}
        
        # 如果当前没有映射，生成随机映射
        if not current_mapping:
            current_mapping = self.generate_random_initial_mapping()
            # print("random mapping:",current_mapping)

        # 2. 计算初始 Cost
        current_cost = self._get_total_cost(current_mapping)
        best_mapping = copy.deepcopy(current_mapping)
        best_cost = current_cost
        
        no_improve = 0
        loops_time = 0
        
        # 3. 爬山搜索 loop
        for i in range(max_iter):
            loops_time += 1# 计数器
            if no_improve >= patience:
                break
            
            candidate_mapping = current_mapping.copy()
            r = random.random()
            
            # --- 动作选择: Move (50%) vs Swap (50%) ---
            if r < 0.5:
                # Move: 将一个逻辑比特移到空闲物理节点
                occupied = set(candidate_mapping.values())
                all_nodes = set(range(self.numNodes))
                free_nodes = list(all_nodes - occupied)
                
                if free_nodes:
                    l_q = random.choice(list(candidate_mapping.keys()))
                    p_target = random.choice(free_nodes)
                    candidate_mapping[l_q] = p_target
                else:
                    # 满载，无法移动，跳过
                    continue
            else:
                # Swap: 交换两个逻辑比特的位置
                keys = list(candidate_mapping.keys())
                if len(keys) >= 2:
                    qa, qb = random.sample(keys, 2)
                    candidate_mapping[qa], candidate_mapping[qb] = candidate_mapping[qb], candidate_mapping[qa]
                else:
                    continue

            # 4. 评估新 Cost
            new_cost = self._get_total_cost(candidate_mapping)
            delta = new_cost - current_cost
            # print("Delta:",delta)
            
            # 5. 贪心接受 (Greedy Accept)
            if delta < 0:
                current_mapping = candidate_mapping
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_mapping = copy.deepcopy(current_mapping)
                    no_improve = 0
            else:
                no_improve += 1
        
        print(f"Running: ",loops_time," times and DSF_mapping finished. Best Cost: {best_cost:.4f}")
        print("BEST MAPPING:",best_mapping)
        
        # 6. 将最优结果应用回类本身
        # self._apply_mapping(best_mapping)
        return best_mapping


# === 新增：生成 GHZ 态 ===
    def generate_GHZ_triplet(self, box1, box2, box3):
        if len(self.GHZ_pool) == 0:
            raise Exception("No more GHZ IDs available.")
        # 简单的验证：确保这三个位置没有被占用，或者是通过融合EPR生成的（具体逻辑取决于你的物理设定）
        # 这里假设直接在三个空位或者特定条件下生成
        
        ghz_id = self.GHZ_pool.pop(0)
        
        # 更新映射
        self.GHZ_ball_to_box[ghz_id] = [box1, box2, box3]
        self.GHZ_box_to_ball[box1] = ghz_id
        self.GHZ_box_to_ball[box2] = ghz_id
        self.GHZ_box_to_ball[box3] = ghz_id
        self.GHZ_triplets[ghz_id] = [box1, box2, box3]
        return ghz_id

    # === 新增：销毁 GHZ 态 ===
    def destroy_GHZ_triplet(self, ghz_id):
        if ghz_id not in self.GHZ_triplets:
            raise Exception(f"No GHZ triplet with ID {ghz_id} exists.")
        
        # 获取这三个物理位置
        boxes = self.GHZ_triplets.pop(ghz_id)
        
        # 清除映射
        self.GHZ_ball_to_box.pop(ghz_id)
        for box in boxes:
            if box in self.GHZ_box_to_ball:
                self.GHZ_box_to_ball.pop(box)
        
        # 归还 ID
        self.GHZ_pool.append(ghz_id)
        self.GHZ_pool.sort()
        return boxes
