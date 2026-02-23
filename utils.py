import numpy as np
import torch
import os
import random
import gym

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(seed)

def calculate_gate_errors(t1, t2, tg):
    """
    计算给定相干时间和门时间下的错误率指标
    
    参数:
    t1: 弛豫时间 (us)
    t2: 去相位时间 (us)
    tg: 门操作持续时间 (us)
    
    返回:
    包含弛豫错误率、去相位错误率以及总错误率的字典
    """
    # 1. 弛豫错误率 (Relaxation error / Damping probability gamma)
    # 对应论文中的公式 (5) [cite: 52]
    p_err1 = 1 - np.exp(-tg / t1)
    
    # 2. 去相位错误率 (Dephasing error)
    # 基于用户提供的公式
    p_err2 = 1 - np.exp(-tg / t2)
    
    # 3. 综合去极化错误率 (Depolarizing error)
    # 对应论文中的公式 (10)，描述了 APD 通道在 Clifford Twirl 下的等效错误率 
    p_total = 0.75 - 0.25 * np.exp(-tg / t1) - 0.5 * np.exp(-tg / t2)
    
    return {
        "p_relaxation": p_err1,
        "p_dephasing": p_err2,
        "p_total_depolarizing": p_total
    }