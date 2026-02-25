import hydra
from omegaconf import DictConfig, OmegaConf

# @hydra.main(version_base=None, config_path="conf", config_name="config")
# class Constants:
#     # EPR GHZ 数量限制
#     MAX_EPR_PAIR = 9
#     MAX_GHZ_PAIR = 9

#     # COOLDOWN时间设置，单位是step
#     COOLDOWN_SWAP = 3
#     COOLDOWN_GENERATE = 5
#     COOLDOWN_GHZGENERATE = 5
#     COOLDOWN_SCORE = 1
#     COOLDOWN_TELE_GATE = 5
#     COOLDOWN_TELE_QUBIT = 5
#     COOLDOWN_TELE_TOFFOLI = 5

#     # entanglement 成功率
#     ENTANGLEMENT_PROBABILITY = 0.95 

#     # 奖励设置
#     REWARD_STOP = -2 
#     REWARD_FOR_SWAP = 0
#     REWARD_SCORE = 50  
#     REWARD_SCORE_TOFFOLI = 60 

#     # 电路规模设置
#     NUMQ = 10  # number of qubits for the random dag/circuit
#     NUMG = 10  # number of gates for the random dag/circuit
#     TOFOLI_GATE_PROB = 0.2

#     # 对于空DAG的奖励(即电路执行完毕)
#     REWARD_EMPTY_DAG = 300 
#     # 奖励值最低
#     REWARD_DEADLINE = -300

#     # 距离设置
#     DISTANCE_MULT = 18  # the multiplier for the difference of total distances metric
#     DISTANCE_QUANTUM_LINK = 30 # virtual distance for quantum links (cross processor)
#     DISTANCE_BETWEEN_EPR = 1

#     # QPU_Type = ["Guadalupe","PenguinV3","PenguinV4"] # ...
#     QPU_Type = ["Guadalupe","Guadalupe","Guadalupe"] # ...
#     # 是否启用Toffoli门，启用后环境中会包含Toffoli门，代理需要学习如何处理
#     IF_ENABLE_TOFFOLIOS = True
#     # 保存的文件名
#     result_path = "running_result/run_on_10Q_10G_with_tofoli02"

#     USE_DEC_DEDLINE = False
#     USE_DSF_MAPPING = False

#     # 退相干截止时间
#     COMPLETION_DEADLINE = 1500

class Constants:
    pass

@hydra.main(version_base=None, config_path="conf", config_name="config")
def load(cfg: DictConfig):
    # 动态将 cfg 中的键值对映射到 Constants 类上，兼容您的旧代码
    for key, value in cfg.items():
        setattr(Constants, key, value)
    
    # 现在您之前的代码依然可以用 Constants.DISTANCE_QUANTUM_LINK 访问
