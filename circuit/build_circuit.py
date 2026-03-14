from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CDKMRippleCarryAdder, HRSCumulativeMultiplier

def analyze_circuit(qc, name):
    """
    分解高级逻辑块，保留基础门 (CX, CCX 和单比特门)，然后统计并打印精准数量
    """
    # 核心秘诀：在 basis_gates 中显式声明 'ccx'
    transpiled_qc = transpile(
        qc, 
        basis_gates=['u', 'p', 'cx', 'ccx', 'h', 'x', 'z', 'y', 's', 't', 'sdg', 'tdg'], 
        optimization_level=1
    )
    
    ops = transpiled_qc.count_ops()
    cx_count = ops.get('cx', 0)
    ccx_count = ops.get('ccx', 0)
    
    print(f"================ {name} ================")
    print(f"✅ Total Qubits (Data + Ancilla): {transpiled_qc.num_qubits}")
    print(f"⏳ Circuit Depth: {transpiled_qc.depth()}")
    print(f"🔗 CX (CNOT) Count: {cx_count}")
    print(f"🧱 CCX (Toffoli) Count: {ccx_count}")
    print(f"📊 All Operations: {dict(ops)}")
    print("-" * 50 + "\n")


if __name__ == "__main__":
    print("🚀 开始生成并评估 Benchmark 电路 (强制保留 CCX 架构)...\n")

    # ==========================================
    # 1. 量子加法器 (Quantum Adder)
    # ==========================================
    adder = CDKMRippleCarryAdder(num_state_qubits=10, kind='full')
    analyze_circuit(adder, "Quantum Adder (4-bit A + 4-bit B)")

    # ==========================================
    # 2. Grover 搜索算法核心 (Grover Oracle & Diffusion)
    # 【修复版】：直接调用电路的 mcx 方法构建 V-chain
    # ==========================================
    num_ctrl_qubits = 10 # 6 个控制比特
    num_ancilla_qubits = max(0, num_ctrl_qubits - 2) # v-chain 需要 n-2 个辅助比特
    total_qubits = num_ctrl_qubits + 1 + num_ancilla_qubits
    
    grover_core = QuantumCircuit(total_qubits)
    
    # 准备比特索引列表
    controls = list(range(num_ctrl_qubits))
    target = num_ctrl_qubits
    ancillas = list(range(num_ctrl_qubits + 1, total_qubits))
    
    # 直接使用电路方法，强制以 v-chain 模式展开，生成密集的 CCX 树
    grover_core.mcx(
        control_qubits=controls, 
        target_qubit=target, 
        ancilla_qubits=ancillas, 
        mode='v-chain'
    )
    
    analyze_circuit(grover_core, f"Grover Core Operator ({num_ctrl_qubits}-qubit Control)")

    # ==========================================
    # 3. 整数乘法器 (Integer Multiplier)
    # ==========================================
    try:
        multiplier = HRSCumulativeMultiplier(num_state_qubits=4, num_result_qubits=4)
        analyze_circuit(multiplier, "Integer Multiplier (3-bit x 3-bit)")
    except ImportError:
        print("[提示] 未找到 HRSCumulativeMultiplier，正在生成乘法器部分积核心逻辑...")
        bit_width = 4
        mult_core = QuantumCircuit(bit_width * 3)
        for i in range(bit_width):
            for j in range(bit_width):
                # 乘法器第一步：计算并累加部分积，极其密集的 CCX 网络
                mult_core.ccx(i, bit_width + j, bit_width * 2 + (i+j) % bit_width)
        analyze_circuit(mult_core, f"Integer Multiplier Core (Partial Products for {bit_width}-bit)")