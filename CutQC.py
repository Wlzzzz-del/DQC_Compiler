from QuatumEnvironment import QuantumEvnironment
from QPUClass import QPUConnection
from DAGClass import DAGClass
from QubitMappingClass import QubitMappingClass
from Constants import Constants
from SystemStateClass import SystemStateClass
from copy import deepcopy
# 初始化DQC环境
import math
import networkx as nx
from qiskit import QuantumCircuit
from cutqc.cutqc.main import CutQC

class CutQCAdapter:
    def __init__(self):
        self.my_arch = QPUConnection() # 所有QPU放在一个类里面
        self.my_DAG = DAGClass()
        # self.dag = dag
        # self.qpu_arch = qpu_arch
        self.max_qubits = self.my_arch.get_qpu_list()
        initial_mapping = None
        qm = QubitMappingClass(self.my_arch.qpu_list, self.my_DAG,
                               self.my_arch.numNodes, self.my_DAG.numQubits, 
                               Constants.MAX_EPR_PAIR,Constants.MAX_GHZ_PAIR, 
                               initial_mapping) 
        qiskit_circuit = self.convert_dagclass_to_qiskit_circuit(
            my_dag=qm.dag,
            num_qubits=qm.numQubits)
        cutqc_result = self.run_cutqc_integration(qm, qiskit_circuit, max_cuts_allowed=15)
        print("CutQC class defined successfully.")


    def convert_dagclass_to_qiskit_circuit(self, my_dag, num_qubits):
        """
        将自定义的 DAGClass 转换为 Qiskit 的 QuantumCircuit 对象，以便喂给 CutQC。
        """
        # 1. 初始化一个对应大小的 Qiskit 电路
        qc = QuantumCircuit(num_qubits)
        
        # 2. 必须按照拓扑顺序（门的执行先后顺序）来遍历你的 DAG
        try:
            sorted_nodes = list(nx.topological_sort(my_dag.DAG))
        except nx.NetworkXUnfeasible:
            # 万一图里有环，退回普通遍历
            sorted_nodes = list(my_dag.DAG.nodes)

        # 3. 逐个提取节点并加入到 Qiskit 电路中
        for node in sorted_nodes:
            # 你的节点格式类似于 (q1, q2, layer) 或 (q1, q2, q3, layer)
            # 去掉最后一个 layer 元素，剩下的就是它作用的比特 ID
            involved_qubits = list(node[:-1])
            
            # 根据作用的比特数量，将相应的门映射到 Qiskit 中
            if len(involved_qubits) == 1:
                # 单比特门，用 H 门或 U 门占位（图切分算法只看拓扑连线，门类型不影响切线判定）
                qc.h(involved_qubits[0])
                
            elif len(involved_qubits) == 2:
                # 双比特门，映射为 CX (CNOT) 门
                qc.cx(involved_qubits[0], involved_qubits[1])
                
            elif len(involved_qubits) == 3:
                # 三比特门，映射为 CCX (Toffoli) 门
                qc.ccx(involved_qubits[0], involved_qubits[1], involved_qubits[2])

        return qc

    def run_cutqc_integration(self,qm_instance, circuit, max_cuts_allowed=10):
        """
        外挂适配器：将 QubitMappingClass 的硬件约束自动映射给 CutQC，并执行电路切割。
        
        参数:
            qm_instance: 实例化后的 QubitMappingClass 对象 (包含了 QPU 列表和硬件约束)
            circuit: Qiskit 的 QuantumCircuit 原电路对象
            max_cuts_allowed: 允许的最大切割次数（EPR 消耗上限）
        """
        print("\n[CutQC Adapter] 正在从 QubitMapping 实例中提取硬件约束...")
        
        # ==========================================
        # 步骤 1: 自动提取分布式硬件限制 (Hardware Constraints)
        # ==========================================
        # 1. 目标子电路数量 (num_subcircuits)：正好等于你的 QPU 数量
        num_qpus = len(qm_instance.qpu_list)
        
        # 2. 子电路最大宽度 (max_subcircuit_width)：即每个 QPU 最多能容纳的物理比特数
        # 假设你的 qpu 对象里有 nodes 属性（或者 numNodes）
        try:
            max_qpu_capacity = max([len(qpu.nodes) for qpu in qm_instance.qpu_list])
        except AttributeError:
            # 如果 qpu 对象没有 nodes 属性，可以用类里的总节点数粗略估计
            max_qpu_capacity = math.ceil(qm_instance.numNodes / num_qpus)
            
        print(f"  --> 目标 QPU 数量: {num_qpus}")
        print(f"  --> 每个 QPU 最大比特容量: {max_qpu_capacity}")
        
        # ==========================================
        # 步骤 2: 构建 CutQC 约束字典
        # ==========================================
        # 这里的约束完全是由真实的 QPU 物理架构动态计算出来的
        cutter_constraints = {
            "max_subcircuit_width": max_qpu_capacity,
            "max_subcircuit_cuts": max_cuts_allowed,
            "subcircuit_size_imbalance": 2, # 允许各芯片负载有轻微不平衡
            "max_cuts": max_cuts_allowed,
            "num_subcircuits": [num_qpus],  # 严格指定切分为 QPU 的数量
        }
        
        # ==========================================
        # 步骤 3: 实例化并运行 CutQC
        # ==========================================
        print("\n[CutQC Adapter] 启动 CutQC 引擎...")
        cutqc = CutQC(
            circuit=circuit,
            cutter_constraints=cutter_constraints,
            verbose=True,
        )
        
        try:
            # 1. 寻找最优切割方案 (内部会调用 MIP 求解器)
            cutqc.cut()
            
            # 2. 评估子电路并计算概率 (可选)
            # cutqc.evaluate(num_shots_fn=None)
            
            # 3. 重建最终结果 (根据你的需求决定是否执行)
            # cutqc.build(mem_limit=32, recursion_depth=1)
            
            # 4. 验证切割与重建的保真度
            # cutqc.verify()
            
            print(f"[CutQC Adapter] ✅ 切割成功！")
            if hasattr(cutqc, 'num_recursions'):
                print(f"  --> 递归层数: {cutqc.num_recursions}")
                
            # 返回切割对象，你可以从中提取切点 (cuts) 和子电路 (subcircuits)
            return cutqc
            
        except Exception as e:
            print(f"[CutQC Adapter] ❌ CutQC 引擎运行失败: {e}")
            return None

    def run_compiler(self):
        """Main workflow of the AdaptDQC compiler."""
        # Step 1: Graph simplification
        simplified_tdag = self.graph_simplification(self.my_DAG)

        # Step 2: Graph clustering
        # Returns a list of graphs at different clustering stages (G0, G1, ... GM)
        clustered_graphs = self.graph_clustering(simplified_tdag)

        # Step 3: Reverse tuning
        tuned_cg = self.reverse_tuning(clustered_graphs)

        # Step 4: Solution generation
        dqc_solution = self.solution_generation(tuned_cg)

        return dqc_solution

    def graph_simplification(self, dag):
        """
        Simplifies the TDAG by merging continuous gates.
        Gates with only one forward ancestor are merged into that ancestor.
        """
        simplified_graph = deepcopy(dag) # Assuming dag is a networkx or similar object
        # TODO: Implement gate merging logic 
        # Find gates S where the node only has 1 incoming edge
        # Merge node into its ancestor
        return simplified_graph

    def graph_clustering(self, tdag):
        """
        Iteratively clusters the TDAG in temporal and spatial domains.
        """
        graphs_history = [tdag]
        current_graph = tdag
        
        # Iteratively apply temporal-spatial clustering until chip capacity is reached
        # while current_max_qubits_in_cluster < self.max_qubits:
        #   1. Temporal clustering: TDAG -> SDHG
        #   2. Spatial clustering: SDHG -> CG
        #   3. Save current_graph to graphs_history
        
        return graphs_history

    def reverse_tuning(self, graphs_history):
        """
        Tunes the initial DQC solution based on performance metrics (e.g., latency, communication).
        Uses a Fiduccia-Mattheyses style heuristic.
        """
        # Start from the most clustered graph (Gm) and tune backwards to G0
        current_best_graph = graphs_history[-1]

        for i in range(len(graphs_history) - 1, 0, -1):
            target_graph = graphs_history[i-1]

            # Map clusters from current_best_graph to target_graph
            # Calculate performance change delta for moving nodes
            # If delta > 0 (performance improves), accept the move

            current_best_graph = target_graph # Update with fine-tuned versionA

        return current_best_graph

    def solution_generation(self, fine_tuned_cg):
        """
        Generates the final distributed subcircuits from the fine-tuned Chip-level Graph.
        """
        # Reverse the mapping from the simplification step
        # Reconstruct subcircuits and identify partition positions (GateComm / QubitComm)
        final_subcircuits = [] 
        return final_subcircuits
