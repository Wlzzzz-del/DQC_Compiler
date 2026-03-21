import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from itertools import combinations
import random

from Constants import Constants

#random.seed()  #choose your lucky number here and fix seed

#from QuatumEnvironment_dummy import QuantumEvnironment     #import the distributed quantum computing simulation environment
from QuatumEnvironment import QuantumEvnironment     #import the distributed quantum computing simulation environment

import copy
import csv


from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
        
        

class EnvUpdater(gym.Env):      #gym is an opanAI's environment generator tools. however, we doing most of it ourselves for the distributed qunatum computing game
    environment_name = "distQuantComp"
    def __init__(self, completion_deadline, encoder): 

        self.quantumEnv = QuantumEvnironment(encoder)
        self.state = self.quantumEnv.state   #state at the beginning decided on by the processor and DAG configurations
        self.mask = self.quantumEnv.mask   #mask at the beginning decided on by the processor and DAG configurations, value always 1 if no masking used
        
        self.action_dim = self.quantumEnv.generate_action_size()  #environment provides action space size
        self.action_list = spaces.Discrete(self.action_dim)   #discrete action space
        self.action_space = self.action_list
        

        self.trials = 1
        self.numSteps = self.quantumEnv.my_arch.deadline
        self.numSteps = completion_deadline

        self.stepCount = 0
        self.dummy_stepCount = 0
        self.EpiCount = 0
        self.successfulGames = 0

        self.ent_cost = 0
        self.waiting_time = 0

        self.epiTotalREward = 0
        self.epiTotalREward_raw = 0

        # Reward normalization / clipping settings (可按需修改或从 Constants/配置中读取)
        self.reward_normalize = True
        self.reward_clip = 5.0  # 将归一化后 reward 限幅到 [-reward_clip, reward_clip]; 设为 None 则不裁剪
        self.reward_scale = 1.0  # 归一化后可选的缩放因子
        # EMA 用于在线估计 reward 的均值/方差
        self._r_ema_mean = 0.0
        self._r_ema_var = 1.0
        self._r_ema_alpha = 0.01
        self._r_initialized = False

        # 奖励值记录
        self.reward_filename = Constants.result_path+'rewards.csv'
        with open(self.reward_filename, 'a', newline="") as file:
            writer = csv.writer(file)
            # row = [self.EpiCount, self.epiTotalREward_raw, self.epiTotalREward, self.successfulGames,self.waiting_time, self.ent_cost]
            writer.writerow(["Episode","RawReward","NorReward","SuccessfulGames","WaitingTime","EntanglementCost"])
        
        # 完成时间记录
        self.done_filename = Constants.result_path+'doneTime.csv'    
        with open(self.reward_filename, 'a', newline="") as file2:
            writer2 = csv.writer(file2)
            writer2.writerow(["Episode","Total"])
        
        # self.epr_
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        

    def step(self, action): #this is the main function where each step of the game takes place, i.e. synchroniztion step
        
        reward, new_state, new_mask, successfulDone = self.quantumEnv.RL_step(action)# 从动作获得 奖励值\状态\mask\完成
        self.state = new_state 
        self.mask = new_mask
        steptemp = copy.deepcopy(self.stepCount)
        steptemp_dummy = copy.deepcopy(self.dummy_stepCount)
        
        
        done = self.deadline_monitor(successfulDone) #deadline monitoring
        #print(self.stepCount)
        
        # if reward >= Constants.REWARD_SCORE:
        #     print("a solution at step: ", steptemp)
        #     print("Dummy Step Count: ", steptemp_dummy)
            
        # 在将环境奖励反馈给 agent 之前，可选地对其进行归一化与裁剪以降低训练方差
        raw_reward = reward
        if done and not successfulDone:
            raw_reward += Constants.REWARD_DEADLINE

        # 更新原始 episode 总奖励（用于记录）
        self.epiTotalREward_raw += raw_reward

        # 归一化/裁剪逻辑
        norm_reward = raw_reward
        if self.reward_normalize:
            if not self._r_initialized:
                self._r_ema_mean = raw_reward
                self._r_ema_var = 1.0
                self._r_initialized = True
            # EMA 更新（在线估计 mean & var）
            diff = raw_reward - self._r_ema_mean
            self._r_ema_mean = (1 - self._r_ema_alpha) * self._r_ema_mean + self._r_ema_alpha * raw_reward
            # 使用无偏估计的简单 EMA 方差更新
            self._r_ema_var = (1 - self._r_ema_alpha) * self._r_ema_var + self._r_ema_alpha * (diff * diff)
            denom = (self._r_ema_var ** 0.5) + 1e-8
            norm_reward = (raw_reward - self._r_ema_mean) / denom * self.reward_scale

        if self.reward_clip is not None:
            norm_reward = max(min(norm_reward, self.reward_clip), -self.reward_clip)

        self.waiting_time += self.quantumEnv.state_object.waiting_time
        self.ent_cost += self.quantumEnv.state_object.ent_cost
        self.epiTotalREward += norm_reward
        if successfulDone:
            self.successfulGames += 1
            #print("total_success: ",self.successfulGames)
        
        if done:
            #print("epiCount: ",self.EpiCount)
            self.EpiCount += 1
            #print("solved/done after step number: ", steptemp)
            # EPI是Episode, epitotalReward是该Episode的总奖励值
            # 同时记录 原始总 reward 与 归一化后的总 reward 以便排查

            # Recording here...
            row = [self.EpiCount, self.epiTotalREward_raw, self.epiTotalREward, self.successfulGames,self.waiting_time, self.ent_cost]
            # EPI是Episode, steptemp是该应该完成的轮数
            row2 = [self.EpiCount, steptemp]
            append_list_as_row(self.reward_filename, row)
            append_list_as_row(self.done_filename, row2)
            self.epiTotalREward = 0
            self.epiTotalREward_raw = 0
        
        self.dummy_stepCount += 1
        if action == 0: ### action=0 is always a stop, and that is the only increase in step
            self.stepCount += 1
        
        # print("[DEBUGGING]action: ", action)
        # print("new_state: ", new_state)
        # print("new_mask: ", new_mask)
        # print("[DEBUGGING]reward: ", reward)
        
        # 尝试将reward缩放到更小的范围，看看能不能稳定训练
        return new_state, new_mask, reward, done, {}       #supposed to return new state, reward and done to the learning agent


    def reset(self):
        self.quantumEnv.environment_reset()  
        self.state = self.quantumEnv.state
        self.mask = self.quantumEnv.mask
        #print("reset_was_called")
        return np.array(self.state), np.array(self.mask)


    def deadline_monitor(self, successfulDone):
        if self.numSteps < self.stepCount or successfulDone:   
            done = True
            self.stepCount = 0
            self.dummy_stepCount = 0
        else:
            done = False
        return done
    



