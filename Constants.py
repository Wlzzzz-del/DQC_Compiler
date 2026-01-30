
class Constants:
    MAX_EPR_PAIR = 9
    MAX_GHZ_PAIR = 9

    COOLDOWN_SWAP = 3
    COOLDOWN_GENERATE = 5
    COOLDOWN_GHZGENERATE = 0
    COOLDOWN_SCORE = 1
    COOLDOWN_TELE_GATE = 5
    COOLDOWN_TELE_QUBIT = 5

    ENTANGLEMENT_PROBABILITY = 0.95 

    REWARD_STOP = -20 
    REWARD_FOR_SWAP = 0
    REWARD_SCORE = 500  

    NUMQ = 18  # number of qubits for the random dag/circuit
    NUMG = 30  # number of gates for the random dag/circuit

    # 对于空DAG的奖励
    REWARD_EMPTY_DAG = 3000 
    # 奖励值最低
    REWARD_DEADLINE = -3000


    DISTANCE_MULT = 18  # the multiplier for the difference of total distances metric
    DISTANCE_QUANTUM_LINK = 30 # virtual distance for quantum links (cross processor)
    DISTANCE_BETWEEN_EPR = 1

    # IF_ENABLE_TOFFOLIOS = False
    # 新增的Config
    # 训练1: 三个异构QPU
    # IF_ENABLE_TOFFOLIOS = False
    # result_path = "running_result/run_on_threeqpu/"
    QPU_Type = ["Guadalupe","PenguinV3","PenguinV4"] # ...

    # 注释上面并启用下面的则正常运行
    # 训练2: 三个异构QPU
    # QPU_Type = ["Guadalupe","Guadalupe"] # ...
    IF_ENABLE_TOFFOLIOS = True
    result_path = "running_result/test"
    USE_DEC_DEDLINE = False
    USE_DSF_MAPPING = False