# Hybrid Entanglement Resource Management for Distributed Quantum Computing vir Reinforcement Learning
## Abstract:
Distributed Quantum Computing (DQC) presents a scalable pathway toward building large-scale quantum systems. However, existing DQC compilers predominantly focus on two-
qubit gates, often decomposing multi-qubit gates into extensive sequences of two-qubit operations, which results in deep circuits and degraded fidelity. Moreover, current research relies almost exclusively on Bell states as the sole entanglement resource, severely limiting the efficient execution of remote multi-qubit gates, such as the Toffoli gate. In this work, we introduce a novel approach that leverages both Bell and GHZ states as hybrid entanglement resources to optimize the deployment of large-scale quantum circuits containing Toffoli gates across distributed QPUs. First, we propose the Spatial Non-Uniform Hypergraph (SNUH) to model the complex topological constraints of GHZ and EPR preparation on QPUs. We then formulate the hybrid resource management problem in DQC as a Markov Decision Process (MDP). To address the complexity of this problem, we devise an approximate strategy and develop HeRO-DQC, a heuristic reinforcement learning-based compiler. HeRO-DQC optimizes circuit fidelity and execution latency by efficiently managing hybrid resources and strategically scheduling gates. Finally, we evaluate HeRO-DQC on quantum circuits using realistic quantum computing setups. The results demonstrate that our approach significantly outperforms existing methods in terms of both fidelity and latency.

This repository contains a Compiler for DQC with entanglement resource including GHZ and EPR that can be trained for real-time compilation of quantum circuits, using different Reinforcement Learning (RL) methods. 
This repository reference https://github.com/ppromponas/CompilerDQC
and the basic unconstrained RL methods that serve as a basis for our implementations are obtained from https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch

The learning agent can be trained using different RL-based approach, including DDQN, DQN and PPO. We implement constrained reinforcement learning approach, where certain state information is used in the form of mask, to allow the learning agent to select only the feasible actions. 

**QuantumEnvironment** is an interface between the RL agent and the compiler. It initializes the environment every time is needed - one time at the initialization and once in every episode. Specifically, it creates for every episode (i) the DQC architecture (by creating an instance of **QPUClass()**), and (ii) the circuit to be executed (**DAGClass()**). In the current version of the code, the DQC architecture used is two IBM Q Guadalupe quantum processors connected through a quantum link. Moreover, the **DAGClass()** generates a random circuit with 30 gates. Therefore, the compiler is being trained with a different quantum circuit in every episode. The constants are contained in **Constants**. **QubitMappingClass()** creates mapping objects that treat physical qubits as a box and logical qubits as balls in order to keep the state of the QPU. **SystemStateClass()** creates the state of the DQC environment and models the change of the system after an action has been decided by the learning agent. 


To run the compiler training process, first edit the DRL method and hyperparamters (optional) inside results/DistQuantum.py, and then run:
```
python results/DistQuantum.py
```
Trained Models are saved inside Models/

## Comparing method
Adapt_DQC
Autocomm
CutQC
DQC-M