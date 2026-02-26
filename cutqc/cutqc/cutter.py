from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.converters import circuit_to_dag, dag_to_circuit
import gurobipy as gp
import math, logging
from qiskit import QuantumCircuit, QuantumRegister


class MIP_Model(object):
    def __init__(
        self,
        n_vertices,
        edges,
        vertex_ids,
        id_vertices,
        num_subcircuit,
        max_subcircuit_width,
        max_subcircuit_cuts,
        subcircuit_size_imbalance,
        num_qubits,
        max_cuts,
    ):
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.edges = edges
        self.n_edges = len(edges)
        self.vertex_ids = vertex_ids
        self.id_vertices = id_vertices
        self.num_subcircuit = num_subcircuit
        self.max_subcircuit_width = max_subcircuit_width
        self.max_subcircuit_cuts = max_subcircuit_cuts
        self.subcircuit_size_imbalance = math.sqrt(subcircuit_size_imbalance)
        self.num_qubits = num_qubits
        self.max_cuts = max_cuts

        """
        Count the number of input qubits directly connected to each node
        """
        self.vertex_weight = {}
        for node in self.vertex_ids:
            qargs = node.split(" ")
            num_in_qubits = 0
            for qarg in qargs:
                if int(qarg.split("]")[1]) == 0:
                    num_in_qubits += 1
            self.vertex_weight[node] = num_in_qubits

        self.model = gp.Model(name="cut_searching")
        self.model.params.OutputFlag = 0
        self._add_variables()
        self._add_constraints()

    def _add_variables(self):
        """
        Indicate if a vertex is in some subcircuit
        """
        self.vertex_var = []
        for i in range(self.num_subcircuit):
            subcircuit_y = []
            for j in range(self.n_vertices):
                j_in_i = self.model.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.BINARY)
                subcircuit_y.append(j_in_i)
            self.vertex_var.append(subcircuit_y)

        """
        Indicate if an edge has one and only one vertex in some subcircuit
        """
        self.edge_var = []
        for i in range(self.num_subcircuit):
            subcircuit_x = []
            for j in range(self.n_edges):
                v = self.model.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.BINARY)
                subcircuit_x.append(v)
            self.edge_var.append(subcircuit_x)

        """
        Total number of cuts
        add 0.1 for numerical stability
        """
        self.num_cuts = self.model.addVar(
            lb=0, ub=self.max_cuts + 0.1, vtype=gp.GRB.INTEGER, name="num_cuts"
        )

        self.subcircuit_counter = {}
        for subcircuit in range(self.num_subcircuit):
            self.subcircuit_counter[subcircuit] = {}

            self.subcircuit_counter[subcircuit]["original_input"] = self.model.addVar(
                lb=0.1,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="original_input_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["rho"] = self.model.addVar(
                lb=0,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="rho_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["O"] = self.model.addVar(
                lb=0,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="O_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["d"] = self.model.addVar(
                lb=0.1,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="d_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["size"] = self.model.addVar(
                lb=self.n_vertices
                / self.num_subcircuit
                / self.subcircuit_size_imbalance,
                ub=self.n_vertices
                / self.num_subcircuit
                * self.subcircuit_size_imbalance,
                vtype=gp.GRB.INTEGER,
                name="size_%d" % subcircuit,
            )
            if self.max_subcircuit_cuts is not None:
                self.subcircuit_counter[subcircuit]["num_cuts"] = self.model.addVar(
                    lb=0.1,
                    ub=self.max_subcircuit_cuts,
                    vtype=gp.GRB.INTEGER,
                    name="num_cuts_%d" % subcircuit,
                )
        self.model.update()

    def _add_constraints(self):
        """
        each vertex in exactly one subcircuit
        """
        for v in range(self.n_vertices):
            self.model.addConstr(
                gp.quicksum(
                    self.vertex_var[i][v] for i in range(self.num_subcircuit)
                ) == 1
            )

    # def _add_constraints(self):
    #     """
    #     each vertex in exactly one subcircuit
    #     """
    #     for v in range(self.n_vertices):
    #         self.model.addConstr(
    #             gp.quicksum(
    #                 [self.vertex_var[i][v] for i in range(self.num_subcircuit)]
    #             ),
    #             gp.GRB.EQUAL,
    #             1,
    #         )

        """
        edge_var=1 indicates one and only one vertex of an edge is in subcircuit
        edge_var[subcircuit][edge] = vertex_var[subcircuit][u] XOR vertex_var[subcircuit][v]
        """
        for i in range(self.num_subcircuit):
            for e in range(self.n_edges):
                u, v = self.edges[e]
                u_vertex_var = self.vertex_var[i][u]
                v_vertex_var = self.vertex_var[i][v]
                self.model.addConstr(self.edge_var[i][e] <= u_vertex_var + v_vertex_var)
                self.model.addConstr(self.edge_var[i][e] >= u_vertex_var - v_vertex_var)
                self.model.addConstr(self.edge_var[i][e] >= v_vertex_var - u_vertex_var)
                self.model.addConstr(
                    self.edge_var[i][e] <= 2 - u_vertex_var - v_vertex_var
                )

        """
        Symmetry-breaking constraints
        Force small-numbered vertices into small-numbered subcircuits:
            v0: in subcircuit 0
            v1: in subcircuit_0 or subcircuit_1
            v2: in subcircuit_0 or subcircuit_1 or subcircuit_2
            ...
        """
        for vertex in range(self.num_subcircuit):
            self.model.addConstr(
                gp.quicksum(
                    [
                        self.vertex_var[subcircuit][vertex]
                        for subcircuit in range(vertex + 1)
                    ]
                )
                == 1
            )

        """
        Compute number of cuts
        """
        self.model.addConstr(
            self.num_cuts
            == gp.quicksum(
                [
                    self.edge_var[subcircuit][i]
                    for i in range(self.n_edges)
                    for subcircuit in range(self.num_subcircuit)
                ]
            )
            / 2
        )

        for subcircuit in range(self.num_subcircuit):
            """
            Compute number of different types of qubit in a subcircuit
            """
            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["original_input"]
                == gp.quicksum(
                    [
                        self.vertex_weight[self.id_vertices[i]]
                        * self.vertex_var[subcircuit][i]
                        for i in range(self.n_vertices)
                    ]
                )
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["rho"]
                == gp.quicksum(
                    [
                        self.edge_var[subcircuit][i]
                        * self.vertex_var[subcircuit][self.edges[i][1]]
                        for i in range(self.n_edges)
                    ]
                )
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["O"]
                == gp.quicksum(
                    [
                        self.edge_var[subcircuit][i]
                        * self.vertex_var[subcircuit][self.edges[i][0]]
                        for i in range(self.n_edges)
                    ]
                )
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["d"]
                == self.subcircuit_counter[subcircuit]["original_input"]
                + self.subcircuit_counter[subcircuit]["rho"]
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["size"]
                == gp.quicksum(
                    [self.vertex_var[subcircuit][v] for v in range(self.n_vertices)]
                )
            )

            if self.max_subcircuit_cuts is not None:
                self.model.addConstr(
                    self.subcircuit_counter[subcircuit]["num_cuts"]
                    == self.subcircuit_counter[subcircuit]["rho"]
                    + self.subcircuit_counter[subcircuit]["O"]
                )

        self.model.setObjective(self.num_cuts, gp.GRB.MINIMIZE)
        self.model.update()

    # def check_graph(self, n_vertices, edges):
    #     # 1. edges must include all vertices
    #     # 2. all u,v must be ordered and smaller than n_vertices
    #     vertices = set([i for (i, _) in edges])
    #     vertices |= set([i for (_, i) in edges])
    #     assert vertices == set(range(n_vertices))
    #     for u, v in edges:
    #         assert u < v
    #         assert u < n_vertices

    def check_graph(self, n_vertices, edges):
        # 1. edges must include all vertices (放宽为：允许有孤立顶点，边里的顶点必须是总顶点的子集)
        # 2. all u,v must be ordered and smaller than n_vertices
        vertices = set([i for (i, _) in edges])
        vertices |= set([i for (_, i) in edges])
        
        # 将原来的 == 改为 issubset (子集判断)
        assert vertices.issubset(set(range(n_vertices))), f"Edges contain unknown vertices! {vertices} vs {n_vertices}"
        for u, v in edges:
            assert u < v
            assert u < n_vertices

    def solve(self):
        # logging.info('solving for %d subcircuits'%self.num_subcircuit)
        # logging.info('model has %d variables, %d linear constraints,%d quadratic constraints, %d general constraints'
        # % (self.model.NumVars,self.model.NumConstrs, self.model.NumQConstrs, self.model.NumGenConstrs))
        try:
            self.model.params.threads = 48
            self.model.Params.TimeLimit = 30
            self.model.optimize()
        except (gp.GurobiError, AttributeError, Exception) as e:
            logging.info("Caught: " + e.message)

        if self.model.solcount > 0:
            self.objective = None
            self.subcircuits = []
            self.optimal = self.model.Status == gp.GRB.OPTIMAL
            self.runtime = self.model.Runtime
            self.node_count = self.model.nodecount
            self.mip_gap = self.model.mipgap
            self.objective = self.model.ObjVal

            for i in range(self.num_subcircuit):
                subcircuit = []
                for j in range(self.n_vertices):
                    if abs(self.vertex_var[i][j].x) > 1e-4:
                        subcircuit.append(self.id_vertices[j])
                self.subcircuits.append(subcircuit)
            assert (
                sum([len(subcircuit) for subcircuit in self.subcircuits])
                == self.n_vertices
            )

            cut_edges_idx = []
            cut_edges = []
            for i in range(self.num_subcircuit):
                for j in range(self.n_edges):
                    if abs(self.edge_var[i][j].x) > 1e-4 and j not in cut_edges_idx:
                        cut_edges_idx.append(j)
                        u, v = self.edges[j]
                        cut_edges.append((self.id_vertices[u], self.id_vertices[v]))
            self.cut_edges = cut_edges
            return True
        else:
            # logging.info('Infeasible')
            return False

from qiskit.dagcircuit import DAGOpNode


from qiskit.dagcircuit import DAGOpNode


from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit import Qubit  # 必须引入 Qubit 类型


from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit import Qubit

def read_circ(circuit):
    dag = circuit_to_dag(circuit)
    edges = []
    node_name_ids = {}
    id_node_names = {}
    vertex_ids = {}
    curr_node_id = 0
    qubit_gate_counter = {}
    
    for qubit in dag.qubits:
        qubit_gate_counter[qubit] = 0
        
    # 【核心修复工具】：提取底层不可变 ID，不再使用脆弱的内存地址
    def get_stable_id(node):
        return getattr(node, '_node_id', id(node))

    # 1. 记录所有操作节点
    for vertex in dag.op_nodes():
        name_parts = []
        for arg in vertex.qargs:
            reg_name = getattr(arg, '_register', None)
            reg_name = reg_name.name if reg_name else "q"
            
            name_parts.append("%s[%d]%d" % (
                reg_name,
                arg._index,
                qubit_gate_counter[arg]
            ))
            qubit_gate_counter[arg] += 1
            
        base_name = " ".join(name_parts)
        if not base_name:
            base_name = f"{vertex.name}_{curr_node_id}"

        stable_id = get_stable_id(vertex)
        if stable_id not in vertex_ids:
            vertex_name = base_name
            collision_count = 1
            while vertex_name in node_name_ids:
                vertex_name = f"{base_name}_{collision_count}"
                collision_count += 1
                
            node_name_ids[vertex_name] = curr_node_id
            id_node_names[curr_node_id] = vertex_name
            vertex_ids[stable_id] = curr_node_id
            curr_node_id += 1

    # 2. 遍历图中的边
    for u, v, wire in dag.edges():
        # 只提取真正的门到门的量子比特连线
        if isinstance(u, DAGOpNode) and isinstance(v, DAGOpNode) and isinstance(wire, Qubit):
            u_stable_id = get_stable_id(u)
            v_stable_id = get_stable_id(v)
            
            # 使用稳定 ID 去映射表里查找，彻底告别内存复用引发的张冠李戴
            u_id = vertex_ids.get(u_stable_id)
            v_id = vertex_ids.get(v_stable_id)
            
            if u_id is not None and v_id is not None:
                edges.append((u_id, v_id))

    n_vertices = curr_node_id
    return n_vertices, edges, node_name_ids, id_node_names

# def read_circ(circuit):
#     dag = circuit_to_dag(circuit)
#     edges = []
#     node_name_ids = {}
#     id_node_names = {}
#     vertex_ids = {}
#     curr_node_id = 0
#     qubit_gate_counter = {}
#     for qubit in dag.qubits:
#         qubit_gate_counter[qubit] = 0
#     for vertex in dag.topological_op_nodes():
#         if len(vertex.qargs) != 2:
#             raise Exception("vertex does not have 2 qargs!")

#         arg0, arg1 = vertex.qargs

#         vertex_name = "%s[%d]%d %s[%d]%d" % (
#             arg0._register.name,
#             arg0._index,
#             qubit_gate_counter[arg0],
#             arg1._register.name,
#             arg1._index,
#             qubit_gate_counter[arg1],
#         )
#         qubit_gate_counter[arg0] += 1
#         qubit_gate_counter[arg1] += 1
#         # logging.info(vertex.op.label,vertex_name,curr_node_id)
#         if vertex_name not in node_name_ids and id(vertex) not in vertex_ids:
#             node_name_ids[vertex_name] = curr_node_id
#             id_node_names[curr_node_id] = vertex_name
#             vertex_ids[id(vertex)] = curr_node_id
#             curr_node_id += 1

#     for u, v, _ in dag.edges():
#         if isinstance(u, DAGOpNode) and isinstance(v, DAGOpNode):# 如果uv是dag节点
#             u_id = vertex_ids[id(u)]
#             v_id = vertex_ids[id(v)]
#             edges.append((u_id, v_id))

#     n_vertices = dag.size()
#     return n_vertices, edges, node_name_ids, id_node_names

from qiskit.converters import circuit_to_dag

def cuts_parser(cuts, circ):
    dag = circuit_to_dag(circ)
    positions = []
    
    for position in cuts:
        source, dest = position
        
        # 1. 净化名称：去掉我们在 read_circ 中加的防碰撞后缀 (例如 "_1")
        source_clean = [x.split('_')[0] for x in source.split(" ")]
        dest_clean = [x.split('_')[0] for x in dest.split(" ")]
        
        # 2. 解析比特名和索引 (例如 "q[0]5" -> qubit="q[0]", idx=5)
        source_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in source_clean
        ]
        dest_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in dest_clean
        ]
        
        qubit_cut = []
        dest_idx_on_wire = None
        
        # 3. 寻找 source 和 dest 共享的比特，且 dest 的索引 > source 的索引
        for source_qubit, src_idx in source_qargs:
            for dest_qubit, dst_idx in dest_qargs:
                # 只要索引是递增的 (不再严格限制为 +1)，就说明顺着这条线切
                if source_qubit == dest_qubit and dst_idx > src_idx:
                    qubit_cut.append(source_qubit)
                    dest_idx_on_wire = dst_idx
                    
        if not qubit_cut:
            raise Exception(f"无法在 {source} 和 {dest} 之间找到共享的量子比特线！")
            
        cut_wire_str = qubit_cut[0] # 取找到的第一个共享线
        
        # 4. 将字符串 "q[0]" 映射回 Qiskit 真实的 Qubit 对象
        wire = None
        reg_name = cut_wire_str.split("[")[0]
        reg_idx = int(cut_wire_str.split("[")[1].split("]")[0])
        
        for qubit in circ.qubits:
            # 兼容高低版本的 Qiskit register 名称获取
            q_reg_name = getattr(qubit, '_register', None)
            q_reg_name = q_reg_name.name if q_reg_name else "q"
            
            if q_reg_name == reg_name and qubit._index == reg_idx:
                wire = qubit
                break
                
        if wire is None:
            raise Exception(f"在电路中找不到比特线 {cut_wire_str}")
            
        # 5. 因为现在的 read_circ 记录了所有门，dest_idx_on_wire 就是精准的切割索引！
        all_Q_gate_idx = dest_idx_on_wire
        positions.append((wire, all_Q_gate_idx))
        
    # 将切点按照索引从大到小排序，防止切割时前面的切点改变了后面门的索引
    positions = sorted(positions, reverse=True, key=lambda cut: cut[1])
    return positions

from qiskit.converters import circuit_to_dag

def cuts_parser(cuts, circ):
    dag = circuit_to_dag(circ)
    positions = []
    
    for position in cuts:
        source, dest = position
        
        # 1. 容错解析函数：安全剥离我们在 read_circ 中加的防碰撞后缀 (如 "_1")
        def parse_qargs(node_name):
            qargs = []
            # 用空格分割多个比特
            for x in node_name.split(" "):
                clean_x = x.split('_')[0]
                # 只处理正常的带有 "]" 的比特名称，过滤掉没有参数的幽灵节点
                if "]" in clean_x:
                    parts = clean_x.split("]")
                    qargs.append((parts[0] + "]", int(parts[1])))
            return qargs

        source_qargs = parse_qargs(source)
        dest_qargs = parse_qargs(dest)
        
        qubit_cut = None
        cut_target_idx = None
        
        # 2. 寻找共享的比特线，无视 source 和 dest 传过来的先后顺序
        for q1, idx1 in source_qargs:
            for q2, idx2 in dest_qargs:
                if q1 == q2:  # 只要比特名称匹配
                    qubit_cut = q1
                    # 切点永远位于两者中较晚执行的那个门 (索引更大) 之前
                    cut_target_idx = max(idx1, idx2)
                    break
            if qubit_cut:
                break
                
        # 3. 极端异常处理：打印出到底是什么怪异的边
        if not qubit_cut:
            print(f"\n[Debug] Source 节点: '{source}' -> 解析为: {source_qargs}")
            print(f"[Debug] Dest 节点:   '{dest}' -> 解析为: {dest_qargs}")
            raise Exception(f"无法在 '{source}' 和 '{dest}' 之间找到共享的量子比特线！这可能是一条连接了不同比特的纯经典条件边。")
            
        # 4. 将字符串 "q[0]" 映射回 Qiskit 真实的 Qubit 对象
        wire = None
        reg_name = qubit_cut.split("[")[0]
        reg_idx = int(qubit_cut.split("[")[1].split("]")[0])
        
        for qubit in circ.qubits:
            # 兼容高低版本的 Qiskit
            q_reg_name = getattr(qubit, '_register', None)
            q_reg_name = q_reg_name.name if q_reg_name else "q"
            
            if q_reg_name == reg_name and qubit._index == reg_idx:
                wire = qubit
                break
                
        if wire is None:
            raise Exception(f"在电路中找不到比特线 {qubit_cut}")
            
        # 5. 直接使用计算出的绝对索引
        positions.append((wire, cut_target_idx))
        
    # 从后往前切 (索引从大到小)，保证前面的切断动作不会影响后面门的原有索引
    positions = sorted(positions, reverse=True, key=lambda cut: cut[1])
    return positions
            


# def cuts_parser(cuts, circ):
#     dag = circuit_to_dag(circ)
#     positions = []
#     for position in cuts:
#         source, dest = position
#         source_qargs = [
#             (x.split("]")[0] + "]", int(x.split("]")[1])) for x in source.split(" ")
#         ]
#         dest_qargs = [
#             (x.split("]")[0] + "]", int(x.split("]")[1])) for x in dest.split(" ")
#         ]
#         qubit_cut = []
#         for source_qarg in source_qargs:
#             source_qubit, source_multi_Q_gate_idx = source_qarg
#             for dest_qarg in dest_qargs:
#                 dest_qubit, dest_multi_Q_gate_idx = dest_qarg
#                 if (
#                     source_qubit == dest_qubit
#                     and dest_multi_Q_gate_idx == source_multi_Q_gate_idx + 1
#                 ):
#                     qubit_cut.append(source_qubit)
#         # if len(qubit_cut)>1:
#         #     raise Exception('one cut is cutting on multiple qubits')
#         for x in source.split(" "):
#             if x.split("]")[0] + "]" == qubit_cut[0]:
#                 source_idx = int(x.split("]")[1])
#         for x in dest.split(" "):
#             if x.split("]")[0] + "]" == qubit_cut[0]:
#                 dest_idx = int(x.split("]")[1])
#         multi_Q_gate_idx = max(source_idx, dest_idx)

#         wire = None
#         for qubit in circ.qubits:
#             if qubit._register.name == qubit_cut[0].split("[")[
#                 0
#             ] and qubit._index == int(qubit_cut[0].split("[")[1].split("]")[0]):
#                 wire = qubit
#         tmp = 0
#         all_Q_gate_idx = None
#         for gate_idx, gate in enumerate(
#             list(dag.nodes_on_wire(wire=wire, only_ops=True))
#         ):
#             if len(gate.qargs) > 1:
#                 tmp += 1
#                 if tmp == multi_Q_gate_idx:
#                     all_Q_gate_idx = gate_idx
#         positions.append((wire, all_Q_gate_idx))
#     positions = sorted(positions, reverse=True, key=lambda cut: cut[1])
#     return positions


def subcircuits_parser(subcircuit_gates, circuit):
    """
    Assign the single qubit gates to the closest two-qubit gates

    Returns:
    complete_path_map[input circuit qubit] = [{subcircuit_idx,subcircuit_qubit}]
    """

    def calculate_distance_between_gate(gate_A, gate_B):
        if len(gate_A.split(" ")) >= len(gate_B.split(" ")):
            tmp_gate = gate_A
            gate_A = gate_B
            gate_B = tmp_gate
        distance = float("inf")
        for qarg_A in gate_A.split(" "):
            qubit_A = qarg_A.split("]")[0] + "]"
            qgate_A = int(qarg_A.split("]")[-1])
            for qarg_B in gate_B.split(" "):
                qubit_B = qarg_B.split("]")[0] + "]"
                qgate_B = int(qarg_B.split("]")[-1])
                # logging.info('%s gate %d --> %s gate %d'%(qubit_A,qgate_A,qubit_B,qgate_B))
                if qubit_A == qubit_B:
                    distance = min(distance, abs(qgate_B - qgate_A))
        # logging.info('Distance from %s to %s = %f'%(gate_A,gate_B,distance))
        return distance

    dag = circuit_to_dag(circuit)
    qubit_allGate_depths = {x: 0 for x in circuit.qubits}
    qubit_2qGate_depths = {x: 0 for x in circuit.qubits}
    gate_depth_encodings = {}
    # logging.info('Before translation :',subcircuit_gates,flush=True)
    for op_node in dag.topological_op_nodes():
        gate_depth_encoding = ""
        for qarg in op_node.qargs:
            gate_depth_encoding += "%s[%d]%d " % (
                qarg._register.name,
                qarg._index,
                qubit_allGate_depths[qarg],
            )
        gate_depth_encoding = gate_depth_encoding[:-1]
        gate_depth_encodings[op_node] = gate_depth_encoding
        for qarg in op_node.qargs:
            qubit_allGate_depths[qarg] += 1
        if len(op_node.qargs) == 2:
            MIP_gate_depth_encoding = ""
            for qarg in op_node.qargs:
                MIP_gate_depth_encoding += "%s[%d]%d " % (
                    qarg._register.name,
                    qarg._index,
                    qubit_2qGate_depths[qarg],
                )
                qubit_2qGate_depths[qarg] += 1
            MIP_gate_depth_encoding = MIP_gate_depth_encoding[:-1]
            # logging.info('gate_depth_encoding = %s, MIP_gate_depth_encoding = %s'%(gate_depth_encoding,MIP_gate_depth_encoding))
            for subcircuit_idx in range(len(subcircuit_gates)):
                for gate_idx in range(len(subcircuit_gates[subcircuit_idx])):
                    if (
                        subcircuit_gates[subcircuit_idx][gate_idx]
                        == MIP_gate_depth_encoding
                    ):
                        subcircuit_gates[subcircuit_idx][gate_idx] = gate_depth_encoding
                        break
    # logging.info('After translation :',subcircuit_gates,flush=True)
    subcircuit_op_nodes = {x: [] for x in range(len(subcircuit_gates))}
    subcircuit_sizes = [0 for x in range(len(subcircuit_gates))]
    complete_path_map = {}
    for circuit_qubit in dag.qubits:
        complete_path_map[circuit_qubit] = []
        qubit_ops = dag.nodes_on_wire(wire=circuit_qubit, only_ops=True)
        for qubit_op_idx, qubit_op in enumerate(qubit_ops):
            gate_depth_encoding = gate_depth_encodings[qubit_op]
            nearest_subcircuit_idx = -1
            min_distance = float("inf")
            for subcircuit_idx in range(len(subcircuit_gates)):
                distance = float("inf")
                for gate in subcircuit_gates[subcircuit_idx]:
                    if len(gate.split(" ")) == 1:
                        # Do not compare against single qubit gates
                        continue
                    else:
                        distance = min(
                            distance,
                            calculate_distance_between_gate(
                                gate_A=gate_depth_encoding, gate_B=gate
                            ),
                        )
                # logging.info('Distance from %s to subcircuit %d = %f'%(gate_depth_encoding,subcircuit_idx,distance))
                if distance < min_distance:
                    min_distance = distance
                    nearest_subcircuit_idx = subcircuit_idx
            assert nearest_subcircuit_idx != -1
            path_element = {
                "subcircuit_idx": nearest_subcircuit_idx,
                "subcircuit_qubit": subcircuit_sizes[nearest_subcircuit_idx],
            }
            if (
                len(complete_path_map[circuit_qubit]) == 0
                or nearest_subcircuit_idx
                != complete_path_map[circuit_qubit][-1]["subcircuit_idx"]
            ):
                # logging.info('{} op #{:d} {:s} encoding = {:s}'.format(circuit_qubit,qubit_op_idx,qubit_op.name,gate_depth_encoding),
                # 'belongs in subcircuit %d'%nearest_subcircuit_idx)
                complete_path_map[circuit_qubit].append(path_element)
                subcircuit_sizes[nearest_subcircuit_idx] += 1

            subcircuit_op_nodes[nearest_subcircuit_idx].append(qubit_op)
    for circuit_qubit in complete_path_map:
        # logging.info(circuit_qubit,'-->')
        for path_element in complete_path_map[circuit_qubit]:
            path_element_qubit = QuantumRegister(
                size=subcircuit_sizes[path_element["subcircuit_idx"]], name="q"
            )[path_element["subcircuit_qubit"]]
            path_element["subcircuit_qubit"] = path_element_qubit
            # logging.info(path_element)
    subcircuits = generate_subcircuits(
        subcircuit_op_nodes=subcircuit_op_nodes,
        complete_path_map=complete_path_map,
        subcircuit_sizes=subcircuit_sizes,
        dag=dag,
    )
    return subcircuits, complete_path_map


def generate_subcircuits(subcircuit_op_nodes, complete_path_map, subcircuit_sizes, dag):
    qubit_pointers = {x: 0 for x in complete_path_map}
    subcircuits = [QuantumCircuit(x, name="q") for x in subcircuit_sizes]
    for op_node in dag.topological_op_nodes():
        subcircuit_idx = list(
            filter(
                lambda x: op_node in subcircuit_op_nodes[x], subcircuit_op_nodes.keys()
            )
        )
        assert len(subcircuit_idx) == 1
        subcircuit_idx = subcircuit_idx[0]
        # logging.info('{} belongs in subcircuit {:d}'.format(op_node.qargs,subcircuit_idx))
        subcircuit_qargs = []
        for op_node_qarg in op_node.qargs:
            if (
                complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]][
                    "subcircuit_idx"
                ]
                != subcircuit_idx
            ):
                qubit_pointers[op_node_qarg] += 1
            path_element = complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]]
            assert path_element["subcircuit_idx"] == subcircuit_idx
            subcircuit_qargs.append(path_element["subcircuit_qubit"])
        # logging.info('-->',subcircuit_qargs)
        subcircuits[subcircuit_idx].append(
            instruction=op_node.op, qargs=subcircuit_qargs, cargs=None
        )
    return subcircuits


def circuit_stripping(circuit):
    # Remove all single qubit gates and barriers in the circuit
    dag = circuit_to_dag(circuit)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circuit.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) == 2 and vertex.op.name != "barrier":
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)


def get_pairs(complete_path_map):
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path) > 1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr + 1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    return O_rho_pairs


def get_counter(subcircuits, O_rho_pairs):
    counter = {}
    for subcircuit_idx, subcircuit in enumerate(subcircuits):
        counter[subcircuit_idx] = {
            "effective": subcircuit.num_qubits,
            "rho": 0,
            "O": 0,
            "d": subcircuit.num_qubits,
            "depth": subcircuit.depth(),
            "size": subcircuit.size(),
        }
    for pair in O_rho_pairs:
        O_qubit, rho_qubit = pair
        counter[O_qubit["subcircuit_idx"]]["effective"] -= 1
        counter[O_qubit["subcircuit_idx"]]["O"] += 1
        counter[rho_qubit["subcircuit_idx"]]["rho"] += 1
    return counter


def find_cuts(
    circuit,
    max_subcircuit_width,
    max_cuts,
    num_subcircuits,
    max_subcircuit_cuts,
    subcircuit_size_imbalance,
    verbose,
):
    stripped_circ = circuit_stripping(circuit=circuit)
    n_vertices, edges, vertex_ids, id_vertices = read_circ(circuit=stripped_circ)
    num_qubits = circuit.num_qubits
    cut_solution = {}

    for num_subcircuit in num_subcircuits:
        if (
            num_subcircuit * max_subcircuit_width - (num_subcircuit - 1) < num_qubits
            or num_subcircuit > num_qubits
            or max_cuts + 1 < num_subcircuit
        ):
            if verbose:
                logging.info(f"{num_subcircuit} subcircuits : IMPOSSIBLE")
            continue
        kwargs = dict(
            n_vertices=n_vertices,
            edges=edges,
            vertex_ids=vertex_ids,
            id_vertices=id_vertices,
            num_subcircuit=num_subcircuit,
            max_subcircuit_width=max_subcircuit_width,
            max_subcircuit_cuts=max_subcircuit_cuts,
            subcircuit_size_imbalance=subcircuit_size_imbalance,
            num_qubits=num_qubits,
            max_cuts=max_cuts,
        )

        mip_model = MIP_Model(**kwargs)
        feasible = mip_model.solve()
        if feasible:
            positions = cuts_parser(mip_model.cut_edges, circuit)
            subcircuits, complete_path_map = subcircuits_parser(
                subcircuit_gates=mip_model.subcircuits, circuit=circuit
            )
            O_rho_pairs = get_pairs(complete_path_map=complete_path_map)
            counter = get_counter(subcircuits=subcircuits, O_rho_pairs=O_rho_pairs)

            cut_solution = {
                "subcircuits": subcircuits,
                "complete_path_map": complete_path_map,
                "num_cuts": len(positions),
                "counter": counter,
            }
            break
        elif verbose:
            logging.info("%d subcircuits : NO SOLUTIONS" % (num_subcircuit))
    if verbose and len(cut_solution) > 0:
        logging.info("-" * 20)
        log_cutter_result(
            num_cuts=cut_solution["num_cuts"],
            subcircuits=cut_solution["subcircuits"],
            counter=cut_solution["counter"],
        )

        logging.info("Model objective value = %.2e" % (mip_model.objective))
        logging.info(f"MIP runtime: {mip_model.runtime}")

        if mip_model.optimal:
            logging.info(f"OPTIMAL, MIP gap = {mip_model.mip_gap}")
        else:
            logging.info(f"NOT OPTIMAL, MIP gap = {mip_model.mip_gap}")
        logging.info("-" * 20)
    return cut_solution


def log_cutter_result(num_cuts, subcircuits, counter):
    logging.info("Cutter result:")
    logging.info(f"{len(subcircuits)} subcircuits, {num_cuts} cuts")

    for subcircuit_idx in range(len(subcircuits)):
        subcircuit_info = f"subcircuit {subcircuit_idx}\n"
        subcircuit_info += (
            "\u03C1 qubits = %d, O qubits = %d, width = %d, effective = %d, depth = %d, size = %d\n"
            % (
                counter[subcircuit_idx]["rho"],
                counter[subcircuit_idx]["O"],
                counter[subcircuit_idx]["d"],
                counter[subcircuit_idx]["effective"],
                counter[subcircuit_idx]["depth"],
                counter[subcircuit_idx]["size"],
            )
        )
        subcircuit_info += f"{subcircuits[subcircuit_idx]}"
        logging.info(subcircuit_info)
