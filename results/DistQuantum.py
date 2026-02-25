import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from QuantumEnv import EnvUpdater

from agents.actor_critic_agents.A2C import A2C
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.A3C import A3C
from agents.policy_gradient_agents.PPO import PPO
from agents.policy_gradient_agents.REINFORCE import REINFORCE
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from agents.DQN_agents.DQN import DQN
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets
from utils import seed_everything
import torch
import os
import wandb
from Constants import Constants, load
import hydra
import hydra
from omegaconf import DictConfig
from Constants import Constants  # 导入刚才建的空壳

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. 注入到 Constants (为了兼容原有代码)
    for key, value in cfg.items():
        # 如果你原来只有 Constants 包含大写字母的变量，可以选择性注入
        # 但全部注入也没有坏处
        setattr(Constants, key, value)

    # 2. 全局环境与设备设置
    seed_everything(cfg.seed)  # 使用 yaml 中的 seed
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    torch.multiprocessing.set_start_method('spawn', force=True)

    # 3. 初始化并填充 Config 对象
    config = Config()
    
    # 自动将 cfg 中属于训练运行的配置注入到 config 实例中
    # 排除掉原属于 Constants 的大写变量，保持 config 对象的纯净
    for key, value in cfg.items():
        if key.islower(): 
            # 将 OmegaConf 的 List/Dict 转换为原生 Python List/Dict 以兼容旧代码
            if OmegaConf.is_config(value):
                setattr(config, key, OmegaConf.to_container(value, resolve=True))
            else:
                setattr(config, key, value)

    # 4. 初始化环境 (调用 yaml 中的常量)
    config.environment = EnvUpdater(completion_deadline=cfg.COMPLETION_DEADLINE)

    # 5. 解析要运行的智能体 (根据名字映射到实际的类)
    agent_map = {
        "DQN": DQN,
        # "PPO": PPO, 
        # "DDQN": DDQN,
        # 把您支持的 agent 类在这里注册一下
    }
    
    # 从配置中读取需要运行的 Agent 列表
    selected_agents = [agent_map[agent_name] for agent_name in cfg.AGENTS]

    # 6. 开始训练
    trainer = Trainer(config, selected_agents)
    trainer.run_games_for_agents()

if __name__ == "__main__":
    main()