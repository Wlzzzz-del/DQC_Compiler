
from operator import sub
import networkx as nx
from itertools import combinations, groupby
import numpy as np
import random
import matplotlib.patches as mpatches

import copy

from QPUClass import QPUConnection
from DAGClass import DAGClass
from QubitMappingClass import QubitMappingClass
from Constants import Constants
from SystemStateClass import SystemStateClass




class QuantumEvnironment():
    
    def __init__(self):

        #Initialize the DQC architecture
        self.my_arch = QPUConnection() # 所有QPU放在一个类里面
        #print(my_arch.G)

        # Qubits之间的物理连接
        self.num_links = self.my_arch.numEdgesClassic #numEdgesQuantum                  # num of links connecting the physical qubits in the processor (not including entanglement links) 
        # Qubits之间的entanglement 连接
        self.num_entanglement_links = self.my_arch.numEdgesQuantum   # how many quantum communication links between distributed processors exist
        # Qubits的物理节点数量
        self.physical_qubits = self.my_arch.numNodes                # num of physical processor locations for bit's to exist
        # 总的可分配的EPR数量
        self.max_epr_pairs = Constants.MAX_EPR_PAIR      # set a threshold for the total EPR pairs
        self.max_ghz_pairs = Constants.MAX_GHZ_PAIR

        # 初始化一个随机quantum电路 Initialize the DAG - quantum circuit
        self.my_DAG = DAGClass()
        # 已经被使用的逻辑qubits数量
        self.num_logical_qubits = self.my_DAG.numQubits         # number of logical qubits that the DAG/algorithm is using
        # DAG中门的数量
        self.dag_vals = self.my_DAG.numGates     # number of gates in the DAG 
        
        # 计算动作空间的size
        self.action_size = self.generate_action_size()        
        # 计算状态空间的size
        # 2是因为EPR+GHZ的num node 状态，4是因为有toffoli三元门
        self.state_size = 2*self.physical_qubits +  4*self.dag_vals  # num of physical qubits will serve as state size with values representing the identity of present logical qubit, the cooldown should be learned in that implementation

        # 产生初始的状态
        self.state_object = self.generate_initial_state_object() 
        # 尝试使用自己设计的函数产生状态
        # self.state = np.array(self.state_object.conver_by_NN())  # create the initial state vector from the init state object
        self.state = np.array(self.state_object.convert_self_to_state_vector())  # create the initial state vector from the init state object
        # print("At the beginning, state is: ", self.state)

        self.mask = np.array(self.get_mask())

    #the functions below (except for RL_step will need to be modified to simulate distributed quantum circuits)
        
    #!this function generates inital state based on processor architecture and initial DAG conditions  (each_reset_start)  - note that it generates a state object from the class SystemStateClass 

    def generate_initial_state_object(self):  #where are all logical qubits (identified as integers) located (index correspond to physical qubits)
        # 这个函数将logical qubit映射到classical qubit上并输出初始的状态矩阵
        initial_mapping = None #Assign random mapping
        #initial_mapping = {0: 1, 1: 2, 2: 20, 3: 21, 4:5, 5: 30, 6:19}

        # qubit的分布类,包括EPR的制备、销毁
        qm = QubitMappingClass(self.my_arch.qpu_list, self.my_DAG, self.my_arch.numNodes, self.my_DAG.numQubits, self.max_epr_pairs,self.max_ghz_pairs, initial_mapping)

        # 根据qm描述系统状态,SystemStateClass类为主要接收action和输出state的类
        init_state_object = SystemStateClass(self.my_arch.qpu_list,self.my_arch.G, self.my_DAG, qm)# 状态类 可以转化成vector

        return init_state_object
    
    #!this function generates the action space size based on possible actions that could be taken
    def generate_action_size(self):
        # 计算动作空间大小
        # 每条phy边有两种状态(SWAP\TELEPORT)
        # 每条quan边有两种状态(Gen\GenGHZ)
        action_size = 1 + 2 * self.num_links + 2 * self.num_entanglement_links 
        # print("DEBUGGING:calculating action size:",action_size," num_links:",self.num_links, " num_ent_link",self.num_entanglement_links)
        #1 for cool off, then 2*num_links with num_links for non-entanglement action (swap) and the
        #rest for creating EPR pairs. Note the actions are in {0,1}  
        return action_size
    


    #!this function generates mask based on state object (in the state object class this is updated AFTER every action) (each_time_step)  
    def get_mask(self):  #which of the actions are feasible
        # 计算mask 应该是看哪个动作是可操作的
        return self.state_object.cur_mask
            
    #!after each completion (DAG completion or deadline failure), next game starts with environment/game reset (each_reset_start)       
    def environment_reset(self): #environment reset fn
        # 重新初始化DQC Initialize the DQC architecture
        #self.my_arch = QPUClass() # not needed since we have already initilized and it does not change dynamically - i.e., in this compiler we have specified a DQC architecture but we can generalize

        #Initialize the DAG - quantum circuit
        self.my_DAG = DAGClass()      ##HERE WE WILL CHANGE THE DAG IN EVERY TIME SLOT BUT FOR NOW WE FIX A SINGLE ONE - NOTE THAT THE STATE SPACE WILL CHANGE WITH NONE AT THE END BUT WE WILL HAVE A FIXED MAX GATE NUMBER
        self.num_logical_qubits = self.my_DAG.numQubits         # number of logical qubits that the DAG/algorithm is using
        self.dag_vals = self.my_DAG.numGates                  # number of gates in the DAG 
        

        self.action_size = self.generate_action_size()        
        self.state_size = 2*self.physical_qubits +  4*self.dag_vals  # num of physical qubits will serve as state size with values representing the identity of present logical qubit, the cooldown should be learned in that implementation

        self.state_object = self.generate_initial_state_object() 
        self.state = np.array(self.state_object.convert_self_to_state_vector())  # create the initial state vector from the init state object

        self.mask = np.array(self.get_mask())
        # print("After Reset, state is: ", self.state)
            
    #network update each time (each_time_step)
    def RL_step(self, action_num):   #action will be a single non-negetive integer in range [0, action_size_val]
    
        reward = 0

        # 计算奖励 
        # print("[DEBUGGING],ACTIONNUM:",action_num)
        reward, self.state_object, successfulDone = self.state_object.step_given_action(action_num)

        # 转换状态至NN能够识别的tensor like
        self.state = np.array(self.state_object.convert_self_to_state_vector())  # create the initial state vector from the init state object as the NN requires it
        self.mask = np.array(self.get_mask())
        
        if successfulDone:
            print("When game won, state is: ", self.state)
            print("DAG nodes after removal", self.my_DAG.DAG.nodes)
            print("#################SOLVED!################################")

        return reward, self.state, self.mask, successfulDone
    

    
















        
        









