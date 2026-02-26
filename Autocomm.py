from QuatumEnvironment import QuantumEvnironment
from QPUClass import QPUConnection
from DAGClass import DAGClass
from QubitMappingClass import QubitMappingClass
from Constants import Constants
from SystemStateClass import SystemStateClass
from copy import deepcopy
from autocom.autocomm import autocomm_full
from autocom.gate_util import build_toffoli_gate
# 初始化DQC环境

import networkx as nx

class Autocomm:
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
        print("Autocomm class defined successfully.")
        epr_used, final_latency, compiled_blocks = self.run_compiler_optimization(qm)

    def run_compiler_optimization(self,qm_instance):
        """
        外挂适配器：将实例化的 QubitMappingClass 对象传入，执行 autocomm 编译优化
        
        参数:
            qm_instance: 已经完成初始 mapping (如 DSF_mapping) 的 QubitMappingClass 实例
        返回:
            epr_cnt: 优化后实际需要的 EPR 数量
            all_latency: 优化后的总延迟
            scheduled_blocks: 最终编译和调度好的门块序列
        """
        print("\n[AdaptDQC Compiler] 正在解析 QubitMapping 实例...")
        
        # ==========================================
        # 步骤 1: 将 DAG 转换为 autocomm 支持的 gate_list
        # ==========================================
        gate_list = []
        
        # 对 DAG 进行拓扑排序，保证门的先后依赖关系
        try:
            sorted_nodes = list(nx.topological_sort(qm_instance.dag.DAG))
        except nx.NetworkXUnfeasible:
            # 万一图里有环（正常量子电路不会有），退化为普通遍历
            sorted_nodes = list(qm_instance.dag.DAG.nodes)

        for node in sorted_nodes:
            # 假设 node 格式为 (qubit1, qubit2, ..., layer)
            involved_qubits = list(node[:-1]) 
            
            if len(involved_qubits) == 1:
                # 单比特门，用 H 门或 RZ 占位，评估延迟用
                gate_list.append(["H", involved_qubits])
                
            elif len(involved_qubits) == 2:
                # 双比特门，默认转为 CX
                gate_list.append(["CX", involved_qubits])
                
            elif len(involved_qubits) == 3:
                # ！！！精髓：利用 gate_util 里的函数，将 Toffoli 门硬分解为 1Q/2Q 门序列
                # 这样 autocomm 的对易合并算法就能完美接管它了！
                qa, qb, qc = involved_qubits
                toffoli_decomp = build_toffoli_gate(qa, qb, qc)
                gate_list.extend(toffoli_decomp)

        # ==========================================
        # 步骤 2: 构建逻辑比特到 QPU_ID 的映射
        # ==========================================
        # autocomm 判定是否产生通信的条件是 mapping 映射的值不同。
        # 因为我们只关心是否“跨芯片”，所以这里把物理 node 转化为 QPU_ID。
        logical_to_qpu_mapping = {}
        
        for logical_qb, physical_node in qm_instance.ball_to_box.items():
            if isinstance(logical_qb, int):  # 过滤掉 "EPR-x" 等字符串 key
                qpu_id = qm_instance.node_to_qpu_id.get(physical_node, -1)
                logical_to_qpu_mapping[logical_qb] = qpu_id

        # ==========================================
        # 步骤 3: 启动底层 autocomm 引擎
        # ==========================================
        print("[AdaptDQC Compiler] 启动底层 autocomm 对易、合并与调度引擎...")
        try:
            print(f"gate_list", gate_list)
            epr_cnt, all_latency, scheduled_blocks = autocomm_full(
                gate_list=gate_list,
                qubit_node_mapping=logical_to_qpu_mapping,
                allow_gate_pattern=True,    # 开启 CRZ 合并等门模式优化
                allow_test_merge=True,      # 开启启发式合并测试
                aggregate_iter_cnt=3,       # 聚合迭代深度
                schedule_iter_cnt=2         # 调度优化深度
            )
            
            print(f"[AdaptDQC Compiler] ✅ 编译成功！")
            print(f"  --> 优化后消耗 EPR 总数: {epr_cnt}")
            print(f"  --> 电路整体延迟 (Latency): {all_latency}")
            return epr_cnt, all_latency, scheduled_blocks
            
        except Exception as e:
            print(f"[AdaptDQC Compiler] ❌ 编译引擎遇到异常: {e}")
            return float('inf'), float('inf'), []

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
