# Compiler for Distributed Quantum Computing: a Reinforcement Learning Approach


This repository contains a Compiler for Distributed Quantum Computing (DQC) that can be trained for real-time compilation of quantum circuits, using different Reinforcement Learning (RL) methods. The compiler is based on the paper "Compiler for Distributed Quantum Computing: a Reinforcement Learning Approach".

The learning agent can be trained using different RL-based approach, including DDQN, DQN and PPO. (We are planning on extending this to other DRL approaches such as DuelingDQN and more.) We implement constrained reinforcement learning approach, where certain state information is used in the form of mask, to allow the learning agent to select only the feasible actions. (The basic unconstrained RL methods that serve as a basis for our implementations are obtained from https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch) 

**QuantumEnvironment** is an interface between the RL agent and the compiler. It initializes the environment every time is needed - one time at the initialization and once in every episode. Specifically, it creates for every episode (i) the DQC architecture (by creating an instance of **QPUClass()**), and (ii) the circuit to be executed (**DAGClass()**). In the current version of the code, the DQC architecture used is two IBM Q Guadalupe quantum processors connected through a quantum link. Moreover, the **DAGClass()** generates a random circuit with 30 gates. Therefore, the compiler is being trained with a different quantum circuit in every episode. The constants are contained in **Constants**. **QubitMappingClass()** creates mapping objects that treat physical qubits as a box and logical qubits as balls in order to keep the state of the QPU. **SystemStateClass()** creates the state of the DQC environment and models the change of the system after an action has been decided by the learning agent. 


To run the compiler training process, first edit the DRL method and hyperparamters (optional) inside results/DistQuantum.py, and then run:
```
python results/DistQuantum.py
```
Trained Models are saved inside Models/



