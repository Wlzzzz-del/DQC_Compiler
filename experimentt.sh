# different rate of toffolios
nohup python results/DistQuantum.py -m TOFOLI_GATE_PROB=0.1,0.2,0.3,0.4 hydra/launcher=joblib hydra.launcher.n_jobs=4 &

# different agents
nohup python results/DistQuantum.py -m AGENTS="[DQN]","[PPO]","[DQN,PPO]" hydra/launcher=joblib hydra.launcher.n_jobs=3 &

# different number of qubits and gates
nohup python results/DistQuantum.py -m NUMQ=10,15,20 NUMG=10,20,30 hydra/launcher=joblib 
hydra.launcher.n_jobs=9 &