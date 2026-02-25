
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as mpatches
import copy

from QPUClass import QPUConnection
from DAGClass import DAGClass
from QubitMappingClass import QubitMappingClass
from Constants import Constants

import itertools



# 核心状态类
class SystemStateClass():
    
    def __init__(self, qpu_list, G, my_DAG, qubit_mapping):
        self.epsilon = 1
        self.G = G# QPU的图
        self.my_DAG = my_DAG# 电路执行图
        self.qm = qubit_mapping# 分布图
        self.qpu_list = qpu_list

        # 更新当前能直接执行的gate
        self.update_frontier()

        # 用于计算举例指标的类
        self.distance_metric = self.calculate_distance_metric() # this metric decides the moving reward - what actions did make the qubits that should come together closer?
        # 保存先前的指标
        self.distance_metric_prev = self.distance_metric # keep track of the previous distance to calculate the reward (the difference)
        self.cur_mask = self.calculate_mask()  # initialize the current mask

 
    def is_action_possible(self, link):
        # Now we need to check both nodes involved in the link
        # print("DEBUGGING in is action possbile:",link)
        return self.G.nodes[link[0]]['weight'] == 0 and self.G.nodes[link[1]]['weight'] == 0 


    def reduce_cooldowns(self):
        for node in self.G.nodes:
            if self.G.nodes[node]['weight'] > 0:
                self.G.nodes[node]['weight'] -= 1


    def perform_action(self, action, link):
        # print("执行动作前检查观测状态：")
        # print("GHZ triplets:",self.qm.GHZ_triplets)
        # print("EPR pairs:",self.qm.EPR_pairs)
        # print("DEBUGGING: 当前将执行的LINK:",link," 当前将执行的Action:",action)
        performed_score = False #make it true only when you indeed performed a score
        # check the cd and produce error if not score or stop. Remember scores we do not produce error since they happen automatically

        if (action!="SCORE_TOFFOLI" and action!= "REMOTE_TOFFOLI" and action!="GENERATE_GHZ" and action != "stop" and action != "SCORE" and action != "tele-gate" and (not self.is_action_possible(link))): 
            raise ValueError(f"Action cannot be performed due to cooldown: {action}")

        if action == "GENERATE":
            self.generate(link)
        elif action == "SWAP":
            self.swap(link)
        elif action == "SCORE":
            performed_score = self.score(link)
            print('*************************WE SCORE!!************************')
            # print("state is: ", self.convert_self_to_state_vector())
            self.update_frontier()
        elif action == "SCORE_TOFFOLI":
            performed_score = self.score_toffoli(link)
            # print('*************************WE SCORE TOFFOLI!!************************')
            # print("state is: ", self.convert_self_to_state_vector())
            self.update_frontier()
        elif action == "tele-gate":
            performed_score = self.tele_gate(link)
            print('*************************WE TELEGATE!!************************')
            # print("state is: ", self.convert_self_to_state_vector())
            self.update_frontier()
        elif action == "tele-qubit":
            self.tele_qubit(link)
        elif action == "stop":
            self.stop()
        # NOTE: 新定义的动作
        elif action == "GENERATE_GHZ":
            self.generate_GHZ(link)
        elif action == "REMOTE_TOFFOLI":
        # REMOTE_TOFFOLI 在FILL Match中调用
            print('*************************WE REMOTE_TOFOLIO_GATE!!************************')
            self.remote_tofo(link)
            self.update_frontier()

        else:
            raise ValueError(f"Unknown action: {action}")
        return performed_score


    def generate(self, link):
        if self.G.edges[link]['label'] != "quantum":
            raise ValueError("GENERATE can only be performed on quantum links.")
        if (self.qm.get_ball(link[0]) != None or self.qm.get_ball(link[1]) != None):
            raise ValueError("GENERATE can only be performed on empty link qubits.")
        self.G.nodes[link[0]]['weight'] = Constants.COOLDOWN_GENERATE
        self.G.nodes[link[1]]['weight'] = Constants.COOLDOWN_GENERATE
        if random.random() < Constants.ENTANGLEMENT_PROBABILITY:  
            self.qm.generate_EPR_pair(*link)
    
    def generate_GHZ(self, links):
        # 接受的link是由多条link组成
        # print(links)
        # if links[0]['label']!="quantum" or links[1]['label']!= "quantum":
        if self.G.edges[links[0][:2]]['label'] != "quantum" or self.G.edges[links[1][:2]]['label'] != "quantum":
            raise ValueError("GENERATE can only be performed on quantum links.")
        if (self.qm.get_ghz_ball(links[0][0]) != None or self.qm.get_ghz_ball(links[0][1]) != None or self.qm.get_ghz_ball(links[1][0]) != None or self.qm.get_ghz_ball(links[1][1]) != None):
            raise ValueError("GENERATE can only be performed on empty link qubits.")

        # print("Here is GHZGEN link:", links)
        linka = links[0]
        linkb = links[1]
        if Constants.COOLDOWN_GHZGENERATE:
            self.G.nodes[linka[0]]['weight'] = Constants.COOLDOWN_GHZGENERATE
            self.G.nodes[linka[1]]['weight'] = Constants.COOLDOWN_GHZGENERATE
            self.G.nodes[linkb[0]]['weight'] = Constants.COOLDOWN_GHZGENERATE
            self.G.nodes[linkb[1]]['weight'] = Constants.COOLDOWN_GHZGENERATE

        # 1. 将两条边的所有点放入一个集合，自动去重
        # 结果应该是 {0, 16, 32}，共 3 个元素
        unique_nodes = set(linka[:2]).union(set(linkb[:2]))
        
        # 2. 检查是否确实是 3 个点 (验证它们是否相连且不构成环/重叠)
        if len(unique_nodes) != 3:
            # 如果不是 3 个点，说明这两条边可能没连上，或者完全重合
            # 视情况抛出异常或返回 False
            print(f"[Error] Cannot generate GHZ: Links {linka} and {linkb} do not form a 3-node chain.")
            return False 

        # 3. 转换回列表以便索引
        nodes_list = list(unique_nodes)
        
        # 4. 调用函数
        # 注意：generate_GHZ_triplet 通常不关心顺序，
        # 但如果你关心中间那个点是“中心”，你需要额外逻辑找出公共点。
        # 如果只是生成资源，直接传这三个点即可。
        if random.random() < Constants.ENTANGLEMENT_PROBABILITY:  
            self.qm.generate_GHZ_triplet(nodes_list[0], nodes_list[1], nodes_list[2])

    def swap(self, link):
        if self.G.edges[link]['label'] == "quantum":
            raise ValueError("SWAP cannot be performed on quantum links.")  
        box1, box2 = link
        EPR_flag = True
        GHZ_flag = True
        ball1 = self.qm.get_ball(box1)
        ball2 = self.qm.get_ball(box2)
        ghz_ball1 = self.qm.get_ghz_ball(box1)
        ghz_ball2 = self.qm.get_ghz_ball(box2)

        # Function to handle swapping of EPR pairs and normal balls
        def handle_swap(box_from, box_to, ball):
            if ball is None:
                return
            # Check if ball is part of EPR pairs
            if ball in self.qm.EPR_pairs:
                self.qm.ball_to_box[ball].remove(box_from)
                self.qm.ball_to_box[ball].append(box_to)
                temp_boxes = self.qm.ball_to_box[ball]              
                self.qm.EPR_pairs[ball] = temp_boxes       #update the EPR pairs as well

            else:
                self.qm.ball_to_box[ball] = box_to

        # NOTE：修改成GHZ版本的SWAP
        def handle_ghz_swap(box_from, box_to, ball):
            if ball is None:
                return
            if ball in self.qm.GHZ_triplets:
                self.qm.GHZ_ball_to_box[ball].remove(box_from)
                self.qm.GHZ_ball_to_box[ball].append(box_to)
                temp_boxes = self.qm.GHZ_ball_to_box[ball]
                self.qm.GHZ_triplets[ball] = temp_boxes
            else:
                self.qm.GHZ_ball_to_box[ball] = box_to

        # If both boxes are empty, don't perform a swap
        if (ball1 is None and ball2 is None):
            EPR_flag = False
        if(ghz_ball1 is None and ghz_ball2 is None):
            GHZ_flag = False
            return
        
        if EPR_flag:
            # Temporary store balls before swapping in box_to_ball mapping
            temp_ball1 = ball1
            temp_ball2 = ball2

            if box1 in self.qm.box_to_ball:
                self.qm.box_to_ball.pop(box1)
            if box2 in self.qm.box_to_ball:
                self.qm.box_to_ball.pop(box2)
            
            # Handle swap of ball1 from box1 to box2
            if temp_ball1 is not None:
                self.qm.box_to_ball[box2] = temp_ball1
                handle_swap(box1, box2, temp_ball1)

            # Handle swap of ball2 from box2 to box1
            if temp_ball2 is not None:
                self.qm.box_to_ball[box1] = temp_ball2
                handle_swap(box2, box1, temp_ball2)

            self.G.nodes[link[0]]['weight'] = Constants.COOLDOWN_SWAP
            self.G.nodes[link[1]]['weight'] = Constants.COOLDOWN_SWAP

        if GHZ_flag:
            temp_ball1 = ghz_ball1
            temp_ball2 = ghz_ball2

            if box1 in self.qm.GHZ_box_to_ball:
                self.qm.GHZ_box_to_ball.pop(box1)
            if box2 in self.qm.GHZ_box_to_ball:
                self.qm.GHZ_box_to_ball.pop(box2)
            
            # Handle swap of ball1 from box1 to box2
            if temp_ball1 is not None:
                self.qm.GHZ_box_to_ball[box2] = temp_ball1
                handle_ghz_swap(box1, box2, temp_ball1)

            # Handle swap of ball2 from box2 to box1
            if temp_ball2 is not None:
                self.qm.GHZ_box_to_ball[box1] = temp_ball2
                handle_ghz_swap(box2, box1, temp_ball2)

            self.G.nodes[link[0]]['weight'] = Constants.COOLDOWN_SWAP
            self.G.nodes[link[1]]['weight'] = Constants.COOLDOWN_SWAP
            pass

        return


    def virtual_swap(self, link):
        box2, other_box = link             #other_box has nothing in it here
        ball2 = self.qm.get_ball(box2)
        if ball2 is None:
            return
        # Function to handle swapping of EPR pairs and normal balls
        def handle_virtual_swap(box_from, box_to, ball):
            # Check if ball is part of EPR pairs
            if ball in self.qm.EPR_pairs:
                self.qm.ball_to_box[ball].remove(box_from)
                self.qm.ball_to_box[ball].append(box_to)
                temp_boxes = self.qm.ball_to_box[ball]              
                self.qm.EPR_pairs[ball] = temp_boxes       #update the EPR pairs as well
            else:
                self.qm.ball_to_box[ball] = box_to

        # Temporary store ball before swapping in box_to_ball mapping
        temp_ball2 = ball2
        if box2 in self.qm.box_to_ball:
            self.qm.box_to_ball.pop(box2)

        # Handle swap of ball2 from box2 to other_box
        self.qm.box_to_ball[other_box] = temp_ball2
        handle_virtual_swap(box2, other_box, temp_ball2)


    def score_toffoli(self, links):
        max_cd = 0
        performed_score = False
        link1,link2 = links
        if self.G.edges[link1]['label'] == "quantum" or self.G.edges[link1]['label'] == "quantum":
            raise ValueError("Score_Toffoli cannot be performed on quantum links.")
        unique_boxes = set(link1).union(set(link2))
        unique_boxes = list(unique_boxes)
        if len(unique_boxes) == 3:
            link_ball1 = self.qm.get_ghz_ball(unique_boxes[0])
            link_ball2 = self.qm.get_ghz_ball(unique_boxes[1])
            link_ball3 = self.qm.get_ghz_ball(unique_boxes[2])
        else:
            return 0
        for gate_info in self.frontier:
            involved_qubits = gate_info[:-1]
            if len(involved_qubits) == 3:
                ball1, ball2, ball3 = involved_qubits
                if {ball1, ball2, ball3} == {link_ball1, link_ball2, link_ball3}:
                    self.my_DAG.remove_node(gate_info)
                    max_cd = max(self.G.nodes[link1[0]]['weight'],self.G.nodes[link1[1]]['weight'],self.G.nodes[link2[0]]['weight'],self.G.nodes[link2[1]]['weight'])
                    self.G.nodes[link1[0]]['weight'] = max_cd + Constants.COOLDOWN_SCORE
                    self.G.nodes[link1[1]]['weight'] = max_cd + Constants.COOLDOWN_SCORE 
                    self.G.nodes[link2[0]]['weight'] = max_cd + Constants.COOLDOWN_SCORE 
                    self.G.nodes[link2[1]]['weight'] = max_cd + Constants.COOLDOWN_SCORE 
                    performed_score = True
                    print("-----we scored TOFFOLI:", ball1, ball2, ball3,"----------")
                    return performed_score

        pass

    def score(self, link):
        max_cd = 0
        performed_score = False # Have scored
        
        if self.G.edges[link]['label'] == "quantum":
            raise ValueError("SCORE cannot be performed on quantum links.")
        
        # 获取当前链路上的两个逻辑比特
        link_ball1 = self.qm.get_ball(link[0])
        link_ball2 = self.qm.get_ball(link[1])
        
        # 如果链路上有空位，直接无法 score
        if link_ball1 is None or link_ball2 is None:
             raise ValueError("could not score (empty link).")

        # --- 修复开始：通用遍历 frontier ---
        # 使用 gate_info 接收不定长的元组
        for gate_info in self.frontier:
            
            # 1. 提取涉及的逻辑比特 (切片去掉最后一个 layer 元素)
            involved_qubits = gate_info[:-1]
            
            # 2. 关键判断：score 是针对 link 的，只能处理 2 比特门
            if len(involved_qubits) == 2:
                ball1, ball2 = involved_qubits
                
                # 3. 检查门所需的比特是否就在当前链路上
                # 使用集合比较 {a,b} == {b,a} 来忽略方向
                if {ball1, ball2} == {link_ball1, link_ball2}:
                    
                    # 4. 移除节点
                    # 注意：建议直接传 gate_info 给 remove_node，
                    # 只要你的 remove_node 支持通过比特匹配即可
                    self.my_DAG.remove_node(gate_info) 
                    
                    # 5. 更新冷却时间
                    max_cd = max(self.G.nodes[link[0]]['weight'], self.G.nodes[link[1]]['weight'])
                    self.G.nodes[link[0]]['weight'] = max_cd + Constants.COOLDOWN_SCORE
                    self.G.nodes[link[1]]['weight'] = max_cd + Constants.COOLDOWN_SCORE 
                    
                    performed_score = True
                    print("-----we scored", ball1, ball2,"----------")
                    return performed_score
            
            # 如果是 Toffoli (len=3)，在这里直接忽略
            # 因为三元门不能通过 score(link) 这种二元操作来触发
            
        # --- 修复结束 ---

        if (not performed_score):
            raise ValueError("could not score.")

    
    # def score(self, link):
    #     max_cd = 0
    #     performed_score = False # Have scored
    #     if self.G.edges[link]['label'] == "quantum":
    #         raise ValueError("SCORE cannot be performed on quantum links.")
    #     for (ball1, ball2, _) in self.frontier:
    #         if (ball1, ball2) in [(self.qm.get_ball(link[0]), self.qm.get_ball(link[1])), (self.qm.get_ball(link[1]), self.qm.get_ball(link[0]))]:
    #             #self.topo_order.remove((ball1, ball2, _))
    #             self.my_DAG.remove_node((ball1, ball2)) #it will understand to remove the fist layer that appears 
    #             max_cd = max(self.G.nodes[link[0]]['weight'], self.G.nodes[link[1]]['weight'])
    #             self.G.nodes[link[0]]['weight'] = max_cd + Constants.COOLDOWN_SCORE
    #             self.G.nodes[link[1]]['weight'] = max_cd + Constants.COOLDOWN_SCORE # since scores happen automatically
    #             performed_score = True
    #             print("-----we scored", ball1, ball2,"----------")
    #             return performed_score
    #     if (not performed_score):
    #         raise ValueError("could not score.")

    def stop(self):
        self.reduce_cooldowns()

    def remote_tofo(self, links):
        # self.generate_GHZ(links)
        # 接受的link是由多条link组成

        # 调度进remote_tofo的无效动作有点多
        # print("等待实现remote tofo")
        print("Remote tofo link:",*links)
         # self.generate_GHZ(links)
        flag = False
        box1, box2, box3 = links
        max_cd = 0
        ball1, ball2, ball3 = self.qm.get_ghz_ball(box1), self.qm.get_ghz_ball(box2), self.qm.get_ghz_ball(box3)

        if ball1 not in self.qm.GHZ_triplets or ball2 not in self.qm.GHZ_triplets or ball3 not in self.qm.GHZ_triplets or ball1 != ball2 or ball2!=ball3 or ball3!=ball1:
            return False

        neighbors_ball1 = set(self.qm.get_ghz_ball(neighbor) for neighbor in self.G.neighbors(box1))
        neighbors_ball2 = set(self.qm.get_ghz_ball(neighbor) for neighbor in self.G.neighbors(box2))
        neighbors_ball3 = set(self.qm.get_ghz_ball(neighbor) for neighbor in self.G.neighbors(box3))

        for gate_info in self.frontier: 
            
            # 1. 提取涉及的逻辑比特 (去掉最后一个 layer)
            involved_qubits = gate_info[:-1]

            # 2. 关键过滤：Tele-gate (基于EPR对) 只能处理 2-qubit 门
            # 如果是 Toffoli (len=3)，必须跳过
            if len(involved_qubits) == 3:
                ball1_frontier, ball2_frontier, ball3_frontier = involved_qubits
                # 3. 检查邻居条件：逻辑比特是否分别位于 EPR 对的两端邻居中
                if (
                    # 情况 1: 1->1, 2->2, 3->3
                    (ball1_frontier in neighbors_ball1 and ball2_frontier in neighbors_ball2 and ball3_frontier in neighbors_ball3) or 
                    # 情况 2: 1->1, 2->3, 3->2
                    (ball1_frontier in neighbors_ball1 and ball2_frontier in neighbors_ball3 and ball3_frontier in neighbors_ball2) or 
                    # 情况 3: 1->2, 2->1, 3->3
                    (ball1_frontier in neighbors_ball2 and ball2_frontier in neighbors_ball1 and ball3_frontier in neighbors_ball3) or 
                    # 情况 4: 1->2, 2->3, 3->1
                    (ball1_frontier in neighbors_ball2 and ball2_frontier in neighbors_ball3 and ball3_frontier in neighbors_ball1) or 
                    # 情况 5: 1->3, 2->1, 3->2
                    (ball1_frontier in neighbors_ball3 and ball2_frontier in neighbors_ball1 and ball3_frontier in neighbors_ball2) or 
                    # 情况 6: 1->3, 2->2, 3->1
                    (ball1_frontier in neighbors_ball3 and ball2_frontier in neighbors_ball2 and ball3_frontier in neighbors_ball1)
                ):
                    
                    # --- 执行 Tele-gate ---
                    # 4. 移除 DAG 节点
                    # 建议直接传入 gate_info，让 remove_node 函数去处理匹配
                    self.my_DAG.remove_node(gate_info) 
                    
                    # 5. 销毁 EPR 对 (ball1 是这一对 EPR 的 ID)
                    # self.qm.destroy_EPR_pair(ball1)
                    self.qm.destroy_GHZ_triplet(ball1)
                    # print("GHZ pool:",self.qm.GHZ_triplets)

                    # 6. 获取涉及的物理节点 (Box)
                    box_q1 = self.qm.get_box(ball1_frontier)
                    box_q2 = self.qm.get_box(ball2_frontier)
                    box_q3 = self.qm.get_box(ball3_frontier)
                    
                    # 7. 计算并更新冷却时间
                    # 注意：所有涉及的组件（两个逻辑比特所在的节点 + EPR 对的两个节点）都需要冷却
                    max_cd = max(self.G.nodes[box_q1]['weight'], 
                                 self.G.nodes[box_q2]['weight'], 
                                 self.G.nodes[box_q3]['weight'], 
                                 self.G.nodes[links[0]]['weight'], 
                                 self.G.nodes[links[1]]['weight'],
                                 self.G.nodes[links[2]]['weight'],
                                 )
                    
                    new_weight = max_cd + Constants.COOLDOWN_TELE_TOFFOLI
                    
                    self.G.nodes[box_q1]['weight'] = new_weight
                    self.G.nodes[box_q2]['weight'] = new_weight
                    self.G.nodes[box_q3]['weight'] = new_weight
                    self.G.nodes[links[0]]['weight'] = new_weight
                    self.G.nodes[links[1]]['weight'] = new_weight
                    self.G.nodes[links[2]]['weight'] = new_weight
                    
                    print("-------we telegate", ball1_frontier, ball2_frontier, ball3_frontier,"----------")
                    flag = True
                    return flag

    # in the tele_gate action, note that the link referes to a "virtual" link between EPR pairs
    # gets as input the positions of the EPR pair and scores using any pair of neighbors (if possible)
    def tele_gate(self, link):
        flag = False # Have performed tele-gate
        box1, box2 = link
        max_cd = 0
        ball1, ball2 = self.qm.get_ball(box1), self.qm.get_ball(box2)
        if ball1 not in self.qm.EPR_pairs or ball2 not in self.qm.EPR_pairs or ball1 != ball2:
            raise ValueError("tele-gate can only happen between EPR pairs.")

        neighbors_ball1 = set(self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box1))
        neighbors_ball2 = set(self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box2))

        # 遍历 Frontier 中的所有门
        # 修复：使用 gate_info 接收不定长元组，避免直接解包报错
        for gate_info in self.frontier: 
            
            # 1. 提取涉及的逻辑比特 (去掉最后一个 layer)
            involved_qubits = gate_info[:-1]

            # 2. 关键过滤：Tele-gate (基于EPR对) 只能处理 2-qubit 门
            # 如果是 Toffoli (len=3)，必须跳过
            if len(involved_qubits) == 2:
                ball1_frontier, ball2_frontier = involved_qubits

                # 3. 检查邻居条件：逻辑比特是否分别位于 EPR 对的两端邻居中
                if ((ball1_frontier in neighbors_ball1 and ball2_frontier in neighbors_ball2) or 
                    (ball1_frontier in neighbors_ball2 and ball2_frontier in neighbors_ball1)):
                    
                    # --- 执行 Tele-gate ---

                    # 4. 移除 DAG 节点
                    # 建议直接传入 gate_info，让 remove_node 函数去处理匹配
                    self.my_DAG.remove_node(gate_info) 
                    
                    # 5. 销毁 EPR 对 (ball1 是这一对 EPR 的 ID)
                    self.qm.destroy_EPR_pair(ball1)

                    # 6. 获取涉及的物理节点 (Box)
                    box_q1 = self.qm.get_box(ball1_frontier)
                    box_q2 = self.qm.get_box(ball2_frontier)
                    
                    # 7. 计算并更新冷却时间
                    # 注意：所有涉及的组件（两个逻辑比特所在的节点 + EPR 对的两个节点）都需要冷却
                    max_cd = max(self.G.nodes[box_q1]['weight'], 
                                 self.G.nodes[box_q2]['weight'], 
                                 self.G.nodes[link[0]]['weight'], 
                                 self.G.nodes[link[1]]['weight'])
                    
                    new_weight = max_cd + Constants.COOLDOWN_TELE_GATE
                    
                    self.G.nodes[box_q1]['weight'] = new_weight
                    self.G.nodes[box_q2]['weight'] = new_weight
                    self.G.nodes[link[0]]['weight'] = new_weight
                    self.G.nodes[link[1]]['weight'] = new_weight
                    
                    print("-------we telegate", ball1_frontier, ball2_frontier,"----------")
                    flag = True
                    return flag

        # for (ball1_frontier, ball2_frontier, _) in self.frontier: 
        #     if ((ball1_frontier in neighbors_ball1 and ball2_frontier in neighbors_ball2) or (ball1_frontier in neighbors_ball2 and ball2_frontier in neighbors_ball1)):
        #         #self.topo_order.remove((ball1_frontier, ball2_frontier, _))
        #         self.my_DAG.remove_node((ball1_frontier, ball2_frontier)) #it will understand to remove the fist layer that appears 
        #         #print(ball1)
        #         self.qm.destroy_EPR_pair(ball1)
        #         max_cd = max(self.G.nodes[self.qm.get_box(ball1_frontier)]['weight'], self.G.nodes[self.qm.get_box(ball2_frontier)]['weight'], self.G.nodes[link[0]]['weight'], self.G.nodes[link[1]]['weight'])
        #         self.G.nodes[self.qm.get_box(ball1_frontier)]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE   # since scores are automatic
        #         self.G.nodes[self.qm.get_box(ball2_frontier)]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE
        #         self.G.nodes[link[0]]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE
        #         self.G.nodes[link[1]]['weight'] = max_cd + Constants.COOLDOWN_TELE_GATE
        #         print("-------we telegate", ball1_frontier, ball2_frontier,"----------")
        #         flag = True
        #         return flag
        # if (not flag):
        #     raise ValueError("tele-gate could not be performed.")
        # return flag            

    
    #needs a link between an EPR particle and a non EPR particle. It teleports the nonEPR qubit to the position that the other half EPR is.
    def tele_qubit(self, link):
        box1, box2 = link
        # print("DEBUGING: watch BOXES:",box1,box2)
        # print("box_to_ball:",self.qm.box_to_ball)
        # print("QM.EPR_pairs:",self.qm.EPR_pairs)
        if self.qm.get_ball(box1) not in self.qm.EPR_pairs and self.qm.get_ball(box2) not in self.qm.EPR_pairs:
            raise ValueError("tele-qubit needs a half of EPR pair.")
            # return False
        if self.qm.get_ball(box1) in self.qm.EPR_pairs and self.qm.get_ball(box2) not in self.qm.EPR_pairs:
            print("------Telequbit performed in", self.qm.get_ball(box2), "using", self.qm.get_ball(box1), "------------------")
            EPR_box = copy.deepcopy(self.qm.query_EPR_pair(self.qm.get_ball(box1)))
            EPR_box.remove(box1)
            other_box = EPR_box[0] #where is the other EPR half
            if (self.G.nodes[other_box]['weight'] != 0):     
                raise ValueError("tele-qubit cannot be performed due to cooldown")
            self.qm.destroy_EPR_pair(self.qm.get_ball(box1))
            self.virtual_swap((box2, other_box))    # box2 contains the qubit and other box contains the box of the EPR half
            self.G.nodes[other_box]['weight'] = Constants.COOLDOWN_TELE_QUBIT
        elif self.qm.get_ball(box2) in self.qm.EPR_pairs and self.qm.get_ball(box1) not in self.qm.EPR_pairs:
            print("------Telequbit performed in", self.qm.get_ball(box1), "using", self.qm.get_ball(box2), "------------------")
            EPR_box = copy.deepcopy(self.qm.query_EPR_pair(self.qm.get_ball(box2)))
            EPR_box.remove(box2)
            other_box = EPR_box[0] #where is the other EPR half
            if (self.G.nodes[other_box]['weight'] != 0):     
                raise ValueError("tele-qubit cannot be performed due to cooldown")
            ##print((box1, other_box))
            self.qm.destroy_EPR_pair(self.qm.get_ball(box2))
            self.virtual_swap((box1, other_box))
            self.G.nodes[other_box]['weight'] = Constants.COOLDOWN_TELE_QUBIT
        self.G.nodes[link[0]]['weight'] = Constants.COOLDOWN_TELE_QUBIT
        self.G.nodes[link[1]]['weight'] = Constants.COOLDOWN_TELE_QUBIT  #########SINCE YOU KILLED IT, YOU SHOULD PERFORM THE ACTIONS LATER ON?



    def update_frontier(self):
        # nodes with no incoming edges
        nodes_no_predecessors = set(self.my_DAG.DAG.nodes()) - {node for _, adj_list in self.my_DAG.DAG.adjacency() for node in adj_list.keys()}
        # update frontier
        self.frontier = nodes_no_predecessors



    def calculate_mask(self):  
        #calculates mask - saves the correct true-falls values depending on whether a link/action can be done and provides a vector form with 0s and 1s with the correct order asked by the agent
        # mask确保action能被执行
        mask = []
        for edge in self.G.edges():  #initialize as if every action is possible
            if (self.G.edges[edge]['label'] != "quantum"):
                self.G.edges[edge]['mask_tele_qubit'] = True   
                self.G.edges[edge]['mask_swap'] = True
                self.G.edges[edge]['mask_tofoligate'] = True
            else: 
                self.G.edges[edge]['mask_generate'] = True  
                self.G.edges[edge]['mask_ghz_generate'] = True  

        # first check cooldowns
        # 如果某条边再Cool down时 不可执行任何操作
        for edge in self.G.edges():  
            if not self.is_action_possible(edge): #Action cannot be performed due to cooldown        #####HERE ADD DONT GENERATE IF READHED THE THRESHOLD
                if (self.G.edges[edge]['label'] != "quantum"):
                    self.G.edges[edge]['mask_tele_qubit'] = False   
                    self.G.edges[edge]['mask_swap'] = False
                else: 
                    self.G.edges[edge]['mask_generate'] = False  
                    self.G.edges[edge]['mask_ghz_generate'] = False

        # now check link by link whether swap is available. The following implement a soft constraint to boost the efficiency of learning.
            # specifically, do not swap if both of the qubits are None (do not swap empty boxes)
        # 如果某条边再Cool down时 不可执行任何操作

        for edge in self.G.edges():  
            #Check: do we have empty boxes? then do not swap
            if (self.G.edges[edge]['label'] != "quantum"):
                if (self.qm.get_ball(edge[0]) == None and self.qm.get_ball(edge[1]) == None):
                    self.G.edges[edge]['mask_swap'] = False 
    

        # now check link by link whether tele-qubit is available (hard constraints)
        qedge = []
        for edge in self.G.edges():  
            #Check: telequbit should have EPR's one 1 side, also the other EPR half should not have cooldown positive
            if (self.G.edges[edge]['label'] != "quantum"):
                #check 1 should have half EPR in one side
                if self.qm.get_ball(edge[0]) not in self.qm.EPR_pairs and self.qm.get_ball(edge[1]) not in self.qm.EPR_pairs:  #"tele-qubit needs a half of EPR pair."
                    self.G.edges[edge]['mask_tele_qubit'] = False 
                #check 2 the other EPR half should be also without cooldown  
                if self.qm.get_ball(edge[0]) in self.qm.EPR_pairs and self.qm.get_ball(edge[1]) not in self.qm.EPR_pairs:
                    EPR_box = copy.deepcopy(self.qm.query_EPR_pair(self.qm.get_ball(edge[0])))
                    EPR_box.remove(edge[0])
                    other_box = EPR_box[0] #where is the other EPR half
                    if (self.G.nodes[other_box]['weight'] != 0):     
                        self.G.edges[edge]['mask_tele_qubit'] = False #tele-qubit cannot be performed due to cooldown"
                elif self.qm.get_ball(edge[1]) in self.qm.EPR_pairs and self.qm.get_ball(edge[0]) not in self.qm.EPR_pairs:
                    EPR_box = copy.deepcopy(self.qm.query_EPR_pair(self.qm.get_ball(edge[1])))
                    EPR_box.remove(edge[1])
                    other_box = EPR_box[0] #where is the other EPR half
                    if (self.G.nodes[other_box]['weight'] != 0):     
                        self.G.edges[edge]['mask_tele_qubit'] = False #tele-qubit cannot be performed due to cooldown"
                #check 3 added later - it implements the soft constraint for efficiency and it cannot implement a telequbit to teleport an empty qubit
                if (self.qm.get_ball(edge[0]) == None or self.qm.get_ball(edge[1]) == None):
                    self.G.edges[edge]['mask_tele_qubit'] = False # tele-qubit better not be performed since we teleport empty qubits
    
            else: #Check whether generate is possible
                qedge.append(edge)
                if(not self.is_action_possible(edge)):
                    self.G.edges[edge]['mask_generate'] = False  
                if (self.qm.get_ball(edge[0]) != None or self.qm.get_ball(edge[1]) != None):    #"GENERATE can only be performed on empty link qubits."
                    self.G.edges[edge]['mask_generate'] = False  
                if len(self.qm.EPR_pool) == 0:                                                  #No more EPR IDs available in the pool.
                    self.G.edges[edge]['mask_generate'] = False  

        # NOTE:先注释 判断GHZ的条件
        import itertools
        for edge1, edge2 in itertools.combinations(qedge, 2):
            # 1. 获取两条边的节点集合
            nodes1 = set(edge1)
            nodes2 = set(edge2)
            # 2. 找公共节点 (Intersection)
            common_nodes = nodes1.intersection(nodes2)
            
            # 如果恰好有一个公共节点，说明这两条边相连 (例如 A-B 和 B-C，公共点是 B)
            if len(common_nodes) == 1:
                # 获取所有涉及的三个节点 (Union)
                all_3_nodes = list(nodes1.union(nodes2)) # [A, B, C]
                
                # 3. 检查这三个节点是否都是空的 (Empty)
                # 只要有一个节点上有球 (不为 None)，就不能生成 GHZ
                can_generate_ghz = True
                for node in all_3_nodes:
                    if self.qm.get_ghz_ball(node) is not None:
                        can_generate_ghz = False
                        break
                
        #         # 4. 还需要检查 GHZ ID 池是否为空
                if len(self.qm.GHZ_pool) == 0:
                    can_generate_ghz = False

                if(can_generate_ghz) and self.is_action_possible(edge1) and self.is_action_possible(edge1):
                    self.G.edges[edge1]['mask_ghz_generate'] = True
                    self.G.edges[edge2]['mask_ghz_generate'] = True

                if self.qm.get_ghz_ball(edge1[0]) != None or self.qm.get_ghz_ball(edge1[1]) != None or self.qm.get_ghz_ball(edge2[0]) != None or self.qm.get_ghz_ball(edge2[1]) != None:
                    self.G.edges[edge1]['mask_ghz_generate'] = False
                    self.G.edges[edge2]['mask_ghz_generate'] = False

        # Generate mask! Now mask as a vector of 0s and 1s should be created
        # First handle 'simple' links

        # +SWAP+TELEQUBIT
        for edge in self.G.edges(data=True):  # Include edge data in the iteration
            if edge[2]['label'] != "quantum":  # Check if the label is 'simple'
                mask.append(int(edge[2]['mask_swap']))  # Append 'mask_swap' first
                mask.append(int(edge[2]['mask_tele_qubit']))  # Then append 'mask_tele_qubit'
                # mask.append(int(edge[2]['mask_tofoligate']))  # Then append 'mask_tofoligate'
        # +EPR_GEN
        Q_edge_set = []
        # Then handle 'quantum' links (or any link that is not 'simple')
        for edge in self.G.edges(data=True):  # Reiterate to maintain the order
            if edge[2]['label'] == "quantum":  # Check if the label is quantum
                mask.append(int(edge[2].get('mask_generate', False)))  # Append 'mask_generate'
                # mask.append(int(edge[2].get('mask_ghz_generate', False)))  # Append 'mask_ghz_generate'
                Q_edge_set.append(edge)

        # +GHZ_GEN
        import itertools
        for edge_a, edge_b in itertools.combinations(Q_edge_set, 2):
            mask.append(int(edge_a[2].get('mask_ghz_generate', False) and edge_b[2].get('mask_ghz_generate', False)))
        #The selected edge and action

        mask = [1] + mask # the 1 at the beginning symbolizes the 'stop' action which is always available.
        return mask   


    # created and returns a random action as a vector (e.g., {'edge':(1,2), 'action':SWAP })  #note that from our heuristic design we can only implement SWAP, tele-qubit and generate. 
    def get_random_action(self):     

        # 从符合条件的mask中抽出indices
        # 随机抽取一个indices
        # decode成动作
        # 所以其实问题在于mask的限制没写好
        # Assuming 'mask' is already created as per calculate_mask() function
        # Create a list of indices where mask is 1
        possible_actions_indices = [i for i, x in enumerate(self.cur_mask) if x == 1]
        # Randomly select one index from possible actions
        selected_index = random.choice(possible_actions_indices)
        random_action = self.decode_action_fromNum(selected_index)
        # Now `random_action` contains the randomly selected edge and action type
        return random_action

    def generate_random_actions_debug(self):
        random_action = 0
        possible_actions_indices = [i for i, x in enumerate(self.cur_mask) if x == 1]
        random_action = random.choice(possible_actions_indices)
        return random_action
    


    def decode_action_fromNum(self, action_num):  #decode a number to the correct action using the self.G.edges command and that first we have swap and then tele-qubit - then we have the generate actions
        # 从数字中编码动作
        # Determine the corresponding edge and action by making the same ordering as how the mask was generated
        edge_action_pairs = []
        # P_edge_set = []
        # important ! 构建动作映射表 这边得加入GHZ

        #### NOTE:这边要和前面mask的顺序对应一致才行!
        for edge in self.G.edges(data=True):
            # print("DEBUGGING: see phy edge info:",edge)
            # DEBUGGING: see phy edge info: (45, 46, {'weight': 0, 'label': 'simple', 'mask_tele_qubit': False, 'mask_swap': False})
            if edge[2]['label'] != "quantum":
                edge_action_pairs.append((edge, 'SWAP'))
                edge_action_pairs.append((edge, 'tele-qubit'))

        Q_edge_set = []
        for edge in self.G.edges(data=True):  
            if edge[2]['label'] == "quantum":  # Check if the label is quantum
                edge_action_pairs.append((edge, 'GENERATE'))
                Q_edge_set.append(edge)

        # 这边列举所有可能的边 添加可能进行GHZ state generation的动作
        import itertools
        for edge_a, edge_b in itertools.combinations(Q_edge_set, 2):
            edge_action_pairs.append(((edge_a,edge_b),"GENERATE_GHZ"))
        #The selected edge and action
        edge_action_pairs = [([], 'stop')] + edge_action_pairs #exacty as how the mask created the action 'stop' - at the beginning

        selected_edge, selected_action = edge_action_pairs[action_num]
        action = {'edge': selected_edge[:2], 'action': selected_action}  # edge is a tuple (u, v), action is a string (we slice the first 2 since the selected edge has also the labels due to data=true argument)
        # print("DEBUGGING:See Action:",action)
        return action 


    




    # step the emulator given a specific action
    def step_given_action(self, action_num):

        matching_scores = [] # here we will store the edges that were picked with the autocomplete method of scoring (scores and tele-gates automatically done after an action)
        reward = 0
        #self.cur_mask = self.calculate_mask() # assume that the mask has been updated before you come here - at the initilization we have the initial mask
    
        # decode为action
        taken_action = self.decode_action_fromNum(action_num)
        cur_state = copy.deepcopy(self)           
        self.perform_action(taken_action['action'], taken_action['edge'])  #make action and change self (state)

        # Fill any scores or tele-gates that can happen immediately after the action of this time slot
        matching_scores = []  #which links were triggered for scores and telegate
        matching_scores,cur_reward = self.fill_matching(matching_scores)   ## Here we auto fill with the scores and tele-gates! The possible scores and tele-gate actually are implemented here automatically!
        reward += cur_reward
        
        
        self.distance_metric = self.calculate_distance_metric() # this metric decides the moving reward - what actions did make the qubits that should come together closer?
        dif_score = 0
        if (reward == 0 and action_num != 0): #it did not score and it is not stop
            dif_score = self.distance_metric_prev - self.distance_metric
            reward = dif_score * Constants.DISTANCE_MULT 
        elif (action_num == 0): #we did stop
            reward = Constants.REWARD_STOP
        self.distance_metric_prev = self.distance_metric #the previous for the next one

    
        self.cur_mask = self.calculate_mask()  # SOS UPDATE the mask with the new changes/new state
        
        flagSuccess = False
        if len(self.my_DAG.DAG.nodes) == 0 : 
            reward = Constants.REWARD_EMPTY_DAG
            flagSuccess = True
        #return reward, self, nx.is_empty(self.my_DAG.DAG)
        return reward, self, flagSuccess
        


    


    #It provides a matching with the possible scores and tele-gates that can happen according to the state.
    def fill_matching(self,matching):
        cur_reward = 0
        # Make a copy of current links in the system excluding those labeled "quantum"

        # 获取所有physical link
        all_links = [link for link in self.G.edges() if self.G.edges[link].get('label') != 'quantum']

        # Iterate over all links for SCORE action - can be done more efficiently by checking the frontier
        # 遍历所有phy link, 挑选可以直接执行的Gate
        # print("watch gate_info:",self.frontier)
        for link in all_links:
            box1, box2 = link# 获取物理比特
            ball1, ball2 = self.qm.get_ball(box1), self.qm.get_ball(box2)# 获取逻辑比特

            for gate_info in self.frontier:# frontier是所有可以立即执行的gate
                # 1. 提取涉及的逻辑比特 (去掉最后一个 layer)
                involved_qubits = gate_info[:-1]

                # 2. 情况 A: 处理二元门 (CNOT/SWAP) - 只有 2 个比特
                if len(involved_qubits) == 2:
                    ball1_frontier, ball2_frontier = involved_qubits
                    
                    # 检查：门需要的两个比特 == 当前链路上的两个比特？
                    if {ball1_frontier, ball2_frontier} == {ball1, ball2}:
                        # Perform the SCORE action and append link to matching
                        self.perform_action('SCORE', link)      
                        cur_reward += Constants.REWARD_SCORE
                        matching.append(link)
                        break 
                
        # 3. 情况 B: 处理三元门 (Toffoli) - 有 3 个比特
        # 注意：Toffoli 不能仅凭一条链路执行，通常需要检查第三个点的连通性
        # 如果你之前没有添加 Toffoli 的特殊处理逻辑，这里可以直接 pass 跳过
        import itertools
        for link1, link2 in itertools.combinations(all_links, 2):
            unique_boxes = set(link1).union(set(link2))
            if len(involved_qubits) == 3:
                three_boxes = list(unique_boxes)
                three_balls = [self.qm.get_ghz_ball(box) for box in three_boxes]
                ball1_frontier, ball2_frontier,ball3_frontier = involved_qubits
                if{ball1_frontier, ball2_frontier, ball3_frontier} == {three_balls[0], three_balls[1], three_balls[2]}:
                    self.perform_action("SCORE_TOFFOLI",[link1,link2])
                    cur_reward += Constants.REWARD_SCORE_TOFFOLI
                    matching.append((link1,link2))
        # Separate loop to iterate over EPR pairs for tele-gate action
        # 遍历所有EPR pairs
        for ball in list(self.qm.EPR_pairs.keys()):
            # 获取包含EPR的box
            link = self.qm.query_EPR_pair(ball)   # get the boxes that contain the EPR pair
            # 有box1，box2
            box1, box2 = link
            neighbors_ball1 = set( self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box1) if  self.G.edges[(box1, neighbor)].get('label') != 'quantum')
            neighbors_ball2 = set(self.qm.get_ball(neighbor) for neighbor in self.G.neighbors(box2) if  self.G.edges[(box2, neighbor)].get('label') != 'quantum') #changed here!!!

            # Iterate over the frontier
            # fix: 使用 gate_info 接收不定长的元组，避免直接解包报错
            for gate_info in self.frontier:

                # 提取涉及的逻辑比特 (去掉最后一个元素 layer)
                involved_qubits = gate_info[:-1]

                if len(involved_qubits) == 2:
                    ball1_frontier, ball2_frontier = involved_qubits
                    # 检查逻辑比特是否位于 EPR 对两端的物理邻居中
                    if ((ball1_frontier in neighbors_ball1 and ball2_frontier in neighbors_ball2) or
                        (ball1_frontier in neighbors_ball2 and ball2_frontier in neighbors_ball1)):
                        self.perform_action('tele-gate', link) 
                        cur_reward += Constants.REWARD_SCORE
                        matching.append(link)
                        break

        for ball in list(self.qm.GHZ_triplets.keys()):
            link = self.qm.query_GHZ_pair(ball)
            box1,box2,box3 = link
            neighbors_ball1 = set(self.qm.get_ghz_ball(neighbor) for neighbor in self.G.neighbors(box1) if  self.G.edges[(box1, neighbor)].get('label') != 'quantum')
            neighbors_ball2 = set(self.qm.get_ghz_ball(neighbor) for neighbor in self.G.neighbors(box2) if  self.G.edges[(box2, neighbor)].get('label') != 'quantum') #changed here!!!
            neighbors_ball3 = set(self.qm.get_ghz_ball(neighbor) for neighbor in self.G.neighbors(box3) if  self.G.edges[(box3, neighbor)].get('label') != 'quantum') #changed here!!!
            for gate_info in self.frontier:
                # 提取涉及的逻辑比特 (去掉最后一个元素 layer)
                involved_qubits = gate_info[:-1]
                if len(involved_qubits) == 3:
                    ball1_frontier, ball2_frontier, ball3_frontier = involved_qubits
                    if (
                        # 情况 1: 1->1, 2->2, 3->3
                        (ball1_frontier in neighbors_ball1 and ball2_frontier in neighbors_ball2 and ball3_frontier in neighbors_ball3) or 
                        # 情况 2: 1->1, 2->3, 3->2
                        (ball1_frontier in neighbors_ball1 and ball2_frontier in neighbors_ball3 and ball3_frontier in neighbors_ball2) or 
                        # 情况 3: 1->2, 2->1, 3->3
                        (ball1_frontier in neighbors_ball2 and ball2_frontier in neighbors_ball1 and ball3_frontier in neighbors_ball3) or 
                        # 情况 4: 1->2, 2->3, 3->1
                        (ball1_frontier in neighbors_ball2 and ball2_frontier in neighbors_ball3 and ball3_frontier in neighbors_ball1) or 
                        # 情况 5: 1->3, 2->1, 3->2
                        (ball1_frontier in neighbors_ball3 and ball2_frontier in neighbors_ball1 and ball3_frontier in neighbors_ball2) or 
                        # 情况 6: 1->3, 2->2, 3->1
                        (ball1_frontier in neighbors_ball3 and ball2_frontier in neighbors_ball2 and ball3_frontier in neighbors_ball1)
                    ):
                    ## 这个判断没有筛选完全
                        # Perform the tele-gate action and append link to matching
                        self.perform_action('REMOTE_TOFFOLI', link) 
                        cur_reward += Constants.REWARD_SCORE_TOFFOLI
                        matching.append(link)
                        break

        return matching, cur_reward

    def calculate_distance_between_balls(self, ball1, ball2, temp_G):
        # 计算某对EPR之间的距离
        # Find the boxes corresponding to ball1 and ball2
        box1 = self.qm.get_box(ball1)
        box2 = self.qm.get_box(ball2)
        epr_links_used = []
        # Calculate the shortest path
        try:
            shortest_path = nx.shortest_path(temp_G, source=box1, target=box2, weight='weight')
            path_length = nx.path_weight(temp_G, shortest_path, weight='weight')
            # Check for EPR links in the path
            for i in range(len(shortest_path) - 1):
                if temp_G.edges[( shortest_path[i], shortest_path[i+1]) ]['virtual'] == True:
                    epr_links_used.append((shortest_path[i], shortest_path[i+1]))
        except nx.NetworkXNoPath:
            # In case there is no path between the two nodes
            path_length = float('inf')

        return path_length, epr_links_used

    def calculate_distance_between_balls(self, ball1, ball2, ball3, temp_G):
        # 计算连接三个GHZ节点所在box的最小网络成本（斯坦纳树）
        box1 = self.qm.get_ghz_box(ball1)
        box2 = self.qm.get_ghz_box(ball2)
        box3 = self.qm.get_ghz_box(ball3)
        
        epr_links_used = []
        terminal_nodes = [box1, box2, box3]

        try:
            # 检查这三个节点是否连通（如果不连通则直接抛出异常或进入except）
            if not (nx.has_path(temp_G, box1, box2) and nx.has_path(temp_G, box1, box3)):
                raise nx.NetworkXNoPath
            
            # 使用斯坦纳树近似算法找到连接这三个节点的最小权重树
            # 返回的是一个包含了所需节点和边的 NetworkX 子图 (Graph 对象)
            steiner_t = nx.approximation.steiner_tree(temp_G, terminal_nodes, weight='weight')
            
            # 计算这棵树的总权重（即连接三个点的总距离）
            path_length = sum(data.get('weight', 1) for u, v, data in steiner_t.edges(data=True))
            
            # 遍历这棵树的所有边，寻找虚拟的EPR链接
            for u, v, data in steiner_t.edges(data=True):
                # 兼容原始图中边的属性
                if data.get('virtual', False) == True:
                    epr_links_used.append((u, v))
                    
        except nx.NetworkXNoPath:
            # 如果这三个节点在网络中无法完全连通
            path_length = float('inf')
            epr_links_used = []

        return path_length, epr_links_used

    def calculate_distance_between_three_balls(self, ball1, ball2, ball3, temp_G):
        # 计算连接三个GHZ节点所在box的最小网络成本（斯坦纳树）
        box1 = self.qm.get_ghz_box(ball1)
        box2 = self.qm.get_ghz_box(ball2)
        box3 = self.qm.get_ghz_box(ball3)
        
        epr_links_used = []
        terminal_nodes = [box1, box2, box3]

        try:
            # 检查这三个节点是否连通（如果不连通则直接抛出异常或进入except）
            if not (nx.has_path(temp_G, box1, box2) and nx.has_path(temp_G, box1, box3)):
                raise nx.NetworkXNoPath
            
            # 使用斯坦纳树近似算法找到连接这三个节点的最小权重树
            # 返回的是一个包含了所需节点和边的 NetworkX 子图 (Graph 对象)
            steiner_t = nx.approximation.steiner_tree(temp_G, terminal_nodes, weight='weight')
            
            # 计算这棵树的总权重（即连接三个点的总距离）
            path_length = sum(data.get('weight', 1) for u, v, data in steiner_t.edges(data=True))
            
            # 遍历这棵树的所有边，寻找虚拟的EPR链接
            for u, v, data in steiner_t.edges(data=True):
                # 兼容原始图中边的属性
                if data.get('virtual', False) == True:
                    epr_links_used.append((u, v))
                    
        except nx.NetworkXNoPath:
            # 如果这三个节点在网络中无法完全连通
            path_length = float('inf')
            epr_links_used = []

        return path_length, epr_links_used


   


    def calculate_distance_metric(self):
        # 计算距离指标
        distance_metric = 0  # Reset the distance metric
        # Create a temporary graph for distance calculation
        temp_G = self.G.copy()

        for edge in temp_G.edges():
            temp_G.edges[edge]['weight'] = 1 # every link will count as distance 1
            temp_G.edges[edge]['virtual'] =  False
            if (temp_G.edges[edge]['label'] == "quantum"): 
                temp_G.edges[edge]['weight'] = Constants.DISTANCE_QUANTUM_LINK  # make it harder to traverse quantum links (they require EPR pair generation conseptually)
        

        # Add links for every EPR pair
        for epr_id, (box1, box2) in self.qm.EPR_pairs.items():
            edge = (box1,box2)
            # Add a "virtual" link
            if (box1,box2) not in self.G.edges:
                temp_G.add_edge(box1, box2, weight = Constants.DISTANCE_BETWEEN_EPR, label="virtual", virtual=True)
            elif (temp_G.edges[edge]['label'] == "quantum"):
                temp_G.edges[edge]['virtual'] = True
                temp_G.edges[edge]['weight'] = Constants.DISTANCE_BETWEEN_EPR # from quantum reduce it temporarily to 1 since we have an entanglement there, remember to increase again when this entanglement is used

        # add ghz virtual links
        # for ghz_id, (box1, box2, box3) in self.qm.GHZ_triplets.items():
        #     edge = (box1,box2,box3)
        #     # temp_G.edges[edge]# debbugging: see if the edge exist
        #     if edge not in self.G.edges:
                    # temp_G.add_edge(edge, weight = Constants.DISTANCE_BETWEEN_EPR, label="virtual", virtual=True)
            # NOTE: GHZ的边是三元边，之前的代码没有正确处理三元边的添加和属性设置，这里需要特别注意
            # elif (temp_G.edges[edge]['label'] == "ghz"):
            #     temp_G.edges[edge]['virtual'] = True
            #     temp_G.edges[edge]['weight'] = Constants.DISTANCE_BETWEEN_EPR # from quantum reduce it temporarily to 1 since we have an entanglement there, remember to increase again when this entanglement is used

        # Iterate over the frontier to calculate distances
        # print("FRONTIER:", self.frontier)
        # for tmp in self.frontier:
        #     if(len(tmp)==2):# EPR的情况
        #         ball1, ball2 = tmp[0], tmp[1]
        #         distance, epr_links_used = self.calculate_distance_between_balls(ball1, ball2, temp_G)
        #         distance_metric += distance
        #         # Remove used EPR links from temp_G
        #         for link in epr_links_used:
        #             if link not in self.G.edges:
        #                 temp_G.remove_edge(*link)
        #             elif (temp_G.edges[link]['label'] == "quantum"):
        #                 temp_G.edges[link]['weight'] = Constants.DISTANCE_QUANTUM_LINK # previous entanglement is used so get it back
        #                 temp_G.edges[link]['virtual'] = False

        #     else:# TOFOLLIO的情况
        #         ball1, ball2, ball3 = tmp[0], tmp[1], tmp[2]
        #         distance1, epr_links_used1 = self.calculate_distance_between_balls(ball1, ball2, temp_G)
        #         distance2, epr_links_used2 = self.calculate_distance_between_balls(ball2, ball3, temp_G)
        #         distance3, epr_links_used3 = self.calculate_distance_between_balls(ball1, ball3, temp_G)
        #         distances_data = [(distance1, epr_links_used1),(distance2, epr_links_used2),(distance3, epr_links_used3)]
        #         distances_data.sort(key=lambda x: x[0], reverse=True)
        #         top1_epr_links = distances_data[0][1]
        #         top2_epr_links = distances_data[1][1]
        #         eprlinks = [top1_epr_links,top2_epr_links]
        #         distance_metric += distances_data[0][0]
        #         distance_metric += distances_data[1][0]

        #         # Remove used EPR links from temp_G
        #         for epr in eprlinks:
        #             for link in epr:
        #                 if link not in self.G.edges:
        #                     # NOTE:这边报错
        #                     temp_G.remove_edge(*link)
        #                 elif (temp_G.edges[link]['label'] == "ghz"):
        #                     temp_G.edges[link]['weight'] = Constants.DISTANCE_QUANTUM_LINK # previous entanglement is used so get it back
        #                     temp_G.edges[link]['virtual'] = False
        for ghz_id, (box1, box2, box3) in self.qm.GHZ_triplets.items():
            # 1. 将三元边拆解为三条两两相连的普通边
            ghz_edges = [(box1, box2), (box2, box3), (box1, box3)]
            
            # 2. 遍历这三条边，分别进行添加或属性修改
            for u, v in ghz_edges:
                # 使用 has_edge 可以完美避免无向图 (A, B) 和 (B, A) 顺序带来的判断错误
                if not self.G.has_edge(u, v):
                    # 如果原拓扑图中根本没有这条边，说明是跨越节点的纯虚拟纠缠
                    temp_G.add_edge(u, v, weight=Constants.DISTANCE_BETWEEN_EPR, label="virtual", virtual=True)
                
                # 修复并激活了您原本注释掉的逻辑：处理底层已经有物理连接的情况
                elif temp_G.has_edge(u, v) and temp_G.edges[u, v].get('label') == "ghz":
                    temp_G.edges[u, v]['virtual'] = True
                    # 从量子链路距离临时缩小为 EPR 纠缠距离 (通常是1)，代表此处存在可用纠缠资源
                    temp_G.edges[u, v]['weight'] = Constants.DISTANCE_BETWEEN_EPR

        for tmp in self.frontier:
            if(len(tmp)==2):# EPR的情况
                ball1, ball2 = tmp[0], tmp[1]
                distance, epr_links_used = self.calculate_distance_between_balls(ball1, ball2, temp_G)
                distance_metric += distance
                # Remove used EPR links from temp_G
                for link in epr_links_used:
                    # 推荐使用 has_edge 替代 in 操作符，自动处理无向图 (u,v) 和 (v,u) 的问题
                    if temp_G.has_edge(*link):
                        if not self.G.has_edge(*link):
                            temp_G.remove_edge(*link)
                        elif temp_G.edges[link].get('label') == "quantum":
                            temp_G.edges[link]['weight'] = Constants.DISTANCE_QUANTUM_LINK # previous entanglement is used so get it back
                            temp_G.edges[link]['virtual'] = False

            else:# TOFFOLI / GHZ 的情况 (3个Ball)
                ball1, ball2, ball3 = tmp[0], tmp[1], tmp[2]
                
                # 直接调用新写的3点斯坦纳树寻径函数
                distance, epr_links_used = self.calculate_distance_between_three_balls(ball1, ball2, ball3, temp_G)
                distance_metric += distance

                # Remove used EPR links from temp_G
                for link in epr_links_used:
                    # 修复报错：先检查 temp_G 中是否确实存在这条边，防止重复删除引发异常
                    if temp_G.has_edge(*link):
                        # 使用 self.G.has_edge(*link) 安全判断边是否存在于原图中
                        if not self.G.has_edge(*link):
                            temp_G.remove_edge(*link)
                        elif temp_G.edges[link].get('label') == "ghz":
                            temp_G.edges[link]['weight'] = Constants.DISTANCE_QUANTUM_LINK
                            temp_G.edges[link]['virtual'] = False

        return distance_metric

    # def convert_self_to_state_vector(self):
    #     # --- 1. 物理映射编码 (my_list) ---
    #     my_list = [-1] * (self.qm.numNodes)
    #     for key, value in self.qm.box_to_ball.items():
    #         if isinstance(value, str):
    #             prefix, num_str = value.split('-')
    #             value = int(num_str) + self.my_DAG.numQubits
    #         my_list[key] = value
    #     # print("STATE: physical mapping:",my_list)

    #     # --- 2. Decoherence time 信息
    #     # 现在采用一个QPU一个退相干时间
    #     dec_time = [-1] * (len(self.qpu_list))
    #     # print(self.G)
    #     for u in range(len(self.qpu_list)):
    #         # print(node)
    #         # print("validation:",self.G.nodes[0])
    #         # print(node)
    #         # NOTE:HERE
    #         dec_time[u] = self.qpu_list[u].dec_time

    #     # print("STATE: Dec_time:",dec_time)

    #     # --- 3. 计算每个逻辑比特的 Urgency ---
    #     node_urgency = [-1] * self.qm.numNodes
        
    #     # 先计算所有逻辑比特的层级
    #     logical_urgency = [999] * self.my_DAG.numQubits
    #     found = [False] * self.my_DAG.numQubits
    #     # for q_i, q_j, layer in self.my_DAG.topo_order:
    #     #     for q in [q_i, q_j]:
    #     #         if not found[q]:
    #     #             logical_urgency[q] = layer
    #     #             found[q] = True
        
    #     for node in self.my_DAG.topo_order:
    #         # node 的结构可能是 (q1, q2, layer) 或 (c1, c2, t, layer)
    #         # 最后一个元素总是 layer
    #         layer = node[-1]
            
    #         # 前面所有的元素都是涉及的量子比特索引
    #         involved_qubits = node[:-1] 

    #         # 更新涉及比特的 urgency
    #         for q in involved_qubits:
    #             if not found[q]:
    #                 logical_urgency[q] = layer
    #                 found[q] = True
            
    #         # 性能优化：如果所有比特都找到了，可以提前退出
    #         if all(found):
    #             break

    #     # 将逻辑比特的紧迫性映射到它当前所在的物理节点上
    #     for phys_node, log_qubit in self.qm.box_to_ball.items():
    #         if isinstance(log_qubit, int): # 只针对普通逻辑比特
    #             node_urgency[phys_node] = logical_urgency[log_qubit]
    #     # print("STATE: Node_urgency:",node_urgency)

    #     state_vector = my_list + node_urgency+dec_time
    #     N = self.qm.numNodes + 3*self.my_DAG.numGates # N is the size of a correct state vector
    #     if len(state_vector) < N:
    #         #print("test")
    #         # 如果state size不够则拼接填充
    #         state_vector.extend([-2] * (N - len(state_vector)))
    #     # print("状态矩阵的大小:",len(state_vector))
    #     return state_vector



    #convert from the actual state class object to the state vector as the RL agent wants it
    def convert_self_to_state_vector(self):
        # 将真实状态矩阵转换成RLstyle格式
        # Initialize a list with None (or a placeholder) for each possible index
        my_list = [-1] * (self.qm.numNodes) # QPU的所有的节点数
        # Populate the list using dictionary keys as indices
        for key, value in self.qm.box_to_ball.items():
            # Instead of EPR-x we now need just the number (i.e., numNodes+x as an index)
            # Check if value is a string
            if isinstance(value, str):
            # Attempt to extract the number part if the format is as expected
                prefix, num_str = value.split('-')
                value = int(num_str) + self.my_DAG.numQubits # Convert the numerical part to an integer (max logical qubit since there does not exist such and after)
            my_list[key] = value# 将用于EPR的QPU 记录为实际value

        my_list2 = [-1] * (self.qm.numNodes) # QPU的所有的节点数
        for key, value in self.qm.GHZ_box_to_ball.items():
            # Instead of EPR-x we now need just the number (i.e., numNodes+x as an index)
            # Check if value is a string
            if isinstance(value, str):
            # Attempt to extract the number part if the format is as expected
                prefix, num_str = value.split('-')
                value = int(num_str) + self.my_DAG.numQubits # Convert the numerical part to an integer (max logical qubit since there does not exist such and after)
            my_list2[key] = value# 将用于EPR的QPU 记录为实际value

        # 将电路图转为拓扑结构插在state vector 后面
        single_numbers_topo_list = [element for tup in self.my_DAG.topo_order for element in tup]  #break (x,y,z) tuple inside topo_order to x,y,z (x,y qubits and z the layer)
        #the above is needed for breaking into the state space vector
        # print("topo_list:",single_numbers_topo_list)
        state_vector = my_list + my_list2 + single_numbers_topo_list
        # 3代表gate的（x,y,z),2代表mylist 和 mylist2
        N = 2*self.qm.numNodes + 4*self.my_DAG.numGates # N is the size of a correct state vector
        if len(state_vector) < N:
            #print("test")
            # 如果state size不够则拼接填充
            state_vector.extend([-2] * (N - len(state_vector)))
        # print("状态矩阵的大小:",len(state_vector))
        return state_vector



    def plot_qubit_mapping(self,pos_G,rew_display, distance_metric_disp, action_disp,topo_disp, frontier_disp):
        G_temp = self.G.copy()
        # Update node labels and color mapping
        labels = {}
        colors = []
        for box in G_temp.nodes:
            weight = G_temp.nodes[box]['weight']  # Get node weight
            if box in self.qm.box_to_ball.keys(): # the box contains a ball
                ball = self.qm.box_to_ball[box]
                if ball in self.qm.EPR_pairs:  # the ball is part of an EPR pair
                    labels[box] = f'{ball} ({weight})'  # Append weight to the label
                    colors.append('cyan')
                else:  # the ball is not part of an EPR pair
                    labels[box] = f'Q-{ball} ({weight})'  # Append weight to the label
                    colors.append('red')
            else:  # the box does not contain a ball
                labels[box] = f'No ({weight})'  # Append weight to the label
                colors.append('green')


        plt.figure(figsize=(10, 8)) 
        # Plot graph
        nx.draw_networkx(G_temp, pos_G, node_color=colors, labels=labels, with_labels=False, font_size=6, node_size=1000)
        nx.draw_networkx_labels(G_temp, pos_G, labels=labels, font_size=9)
        # Show quantum links
        quantum_links = [(u,v) for (u,v,d) in G_temp.edges(data=True) if d['label'] == 'quantum']
        nx.draw_networkx_edges(G_temp, pos_G, edgelist=quantum_links, width=2, alpha=0.5, edge_color='red')

        # Add text at the bottom of the figure
        plt.text(0.5, 1.1, 'reward='+str(rew_display) + ",\n" 'distance_metric='+str(distance_metric_disp) + ",\n" + 'action='+str(action_disp) , ha='center', va='top', transform=plt.gca().transAxes)
        plt.text(0.5, -0.1, 'dag_left='+str(topo_disp) + ",\n"+'frontier='+str(frontier_disp), ha='center', va='bottom', transform=plt.gca().transAxes)
        plt.show()