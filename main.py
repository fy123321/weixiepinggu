# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import copy
import os
import random
import xlsxwriter
import gym
import numpy as np
import torch
from nn.Discrete.memory import Memory
from nn.Discrete.PPO import PPO
from uav_gym.uav_config.action_config import get_blue_action_from_min_max_e, get_blue_single_action, \
    get_blue_action_from_min_max_s
from uav_gym.uav_config.state_config import get_fixed_state, get_random_state, get_fixed_blue_random_red
# from uav_gym.uav_battle_env_continuous_action import UavBattleEnvContinuousAction
from uav_gym.uav_continuous import UavBattleEnvContinuousAction
from uav_gym.uav_discrete import UavBattleEnv
from uav_gym.uav_config.env_config import *
from uav_gym.env_new import Model
from uav_gym.uav_config.action_config import next_state
from uav_gym.reward_custom.reward_e import Reward as Reward
from uav_gym.reward_custom.reward_e import Reward as Reward_e
from uav_gym.reward_custom.reward_superiority import Reward as Reward_s
from uav_gym.uav_config.state_builder_regular import StateBuilderRegular as StateBuilder

N_STEPS = 8
MOD_STEPS = 1


class LogAna:
    def __init__(self):
        self.workbook = xlsxwriter.Workbook('2V1_ang——扩大距离2000.xlsx')  # 建立文件

        self.worksheet = self.workbook.add_worksheet()  # 建立sheet， 可以work.add_worksheet('employee')来指定sheet名，但中文名会报UnicodeDecodeErro的错误
        self.worksheet.write('A1', '步数')
        self.worksheet.write('B1', 'model')
        self.worksheet.write('C1', 'reward')
        self.worksheet.write('D1', 'win_agent')
        # worksheet.write('E1', 'minmax_reward')
        self.worksheet.write('E1', 'model4218')
        # worksheet.write('F1', 'model_reward')
        self.worksheet.write('F1', 'model_e vs diff')
        self.worksheet.write('G1', 'minmax_reward_num')
        self.worksheet.write('H1', 'model_reward_num')
        self.worksheet.write('I1', '蓝方选择')


class Analyse:
    def __init__(self, gamma, r_model1, r_model2, log):
        # self.env = env
        self.GAMMA = gamma
        # self.state_gym = state_gym
        self.red_model1 = r_model1
        self.red_model2 = r_model2
        self.log = log
        self.reward_instance = Reward(None, None, None, None)
        self.FirstStep = True
        self.Red_Model_Choice = None

    def re_init(self):
        '''每一局初始化'''
        self.FirstStep = True
        self.Red_Model_Choice = None

    def step_mulit(self, action, state):
        '''
        step
        :param action:
        :param state:
        :return:
        '''

        red_action_ = action[0]
        blue_action_ = action[1]
        err_msg = "%r (%s) invalid" % (red_action_, type(red_action_))
        assert action_space['Continuous'].contains(red_action_), err_msg
        err_msg = "%r (%s) invalid" % (blue_action_, type(blue_action_))
        assert action_space['Continuous'].contains(blue_action_), err_msg

        state_red_ = state[0]
        state_blue_ = state[1]
        # print('红方状态：', observation_red, '\n蓝方状态：', observation_blue)

        action_vector_red = red_action_
        next_state_red_, _ = next_state(state_red_, action_vector_red)

        action_vector_blue = blue_action_  # self.instance_model.acculturation([self.state[1],self.state[0]])
        next_state_blue_, _ = next_state(state_blue_, action_vector_blue)
        state = [next_state_red_, next_state_blue_]

        done, win_agent, ang_r_2_rb, ang_b_2_rb, dis = self._5step_done(state)
        if done:
            reward_r = DONE_REWARDS[win_agent]
            return None, reward_r, done, ang_r_2_rb, ang_b_2_rb, dis

        self.reward_instance.re_init(state_red_, state_blue_, next_state_red_, next_state_blue_)
        reward = self.reward_instance.get_reward()
        # 这个版本的奖励函数需要传入原始状态与做完动作后的状态进行对比
        # print(state,reward,done)
        return state, reward, done, ang_r_2_rb, ang_b_2_rb, dis

    def _5step_done(self, state):
        done = False
        win_agent = NOT_END

        sta_r, sta_b = state[0], state[1]

        dis = np.linalg.norm(np.array([sta_r[0] - sta_b[0], sta_r[1] - sta_b[1], sta_r[2] - sta_b[2]]))

        vector_red_2_blue = np.array(
            [sta_b[0] - sta_r[0], sta_b[1] - sta_r[1], sta_b[2] - sta_r[2]])  # 以红方为起点到蓝方的向量
        # vector_v_red = np.array(
        #     [[sta_r[3] * math.cos(sta_r[4]) * math.sin(sta_r[5])], [sta_r[3] * math.cos(sta_r[4]) * math.cos(sta_r[5])],
        #      [sta_r[3] * math.sin(sta_r[4])]])  # red x_dot, y_dot, z_dot
        # vector_v_blue = np.array(
        #     [[sta_b[3] * math.cos(sta_b[4]) * math.sin(sta_b[5])], [sta_b[3] * math.cos(sta_b[4]) * math.cos(sta_b[5])],
        #      [sta_b[3] * math.sin(sta_b[4])]])
        # 更改第一分量和第二分量 2021-12-28
        vector_v_red = np.array(
            [[sta_r[3] * math.cos(sta_r[4]) * math.cos(sta_r[5])],
             [sta_r[3] * math.cos(sta_r[4]) * math.sin(sta_r[5])],
             [sta_r[3] * math.sin(sta_r[4])]])  # red x_dot, y_dot, z_dot
        vector_v_blue = np.array(
            [[sta_b[3] * math.cos(sta_b[4]) * math.cos(sta_b[5])],
             [sta_b[3] * math.cos(sta_b[4]) * math.sin(sta_b[5])],
             [sta_b[3] * math.sin(sta_b[4])]])

        # 红方速度向量与红蓝向量的夹角
        ang_r_2_rb = math.acos(
            np.dot(vector_red_2_blue, vector_v_red) / (np.linalg.norm(vector_red_2_blue) * np.linalg.norm(
                vector_v_red)))
        ang_b_2_rb = math.acos(np.dot(vector_red_2_blue, vector_v_blue) / (
                np.linalg.norm(vector_red_2_blue) * np.linalg.norm(vector_v_blue)))

        # print("phi1", phi1, "\nphi2:", phi2)

        # 红方胜的条件
        if dis <= attack_dis_r and abs(ang_r_2_rb) <= attack_ang_r / 2:
            win_agent = RED_WIN
            done = True

        # 蓝方胜的条件：
        if dis <= attack_dis_b and abs(ang_b_2_rb) >= math.pi - attack_ang_b / 2:
            win_agent = BLUE_WIN
            done = True

        # 3.双方同时获胜，条件：双方同时进入对方攻击区
        if dis <= attack_dis_r and dis <= attack_dis_b:
            if abs(ang_r_2_rb) <= attack_ang_r / 2 and ang_b_2_rb >= math.pi - attack_ang_b / 2:
                win_agent = DRAW
                done = True

        # 出界的结束情况
        r_out = not (np.all(sta_r >= env_low) and np.all(sta_r <= env_high))
        b_out = not (np.all(sta_b >= env_low) and np.all(sta_b <= env_high))
        # 同时出界
        if r_out and b_out:
            win_agent = DRAW
            done = True
            # for i in range(2):
            #     for j in range(4):
            #         self.state[i][j] = np.clip(self.state[i][j], self.low[j], self.high[j])

        if r_out:
            win_agent = RED_OUT
            done = True
            # 出界需要修正最后的状态以避免状态超出界限RLLib报错
            # for i in range(4):
            #     self.state[0][i] = np.clip(self.state[0][i], self.low[i], self.high[i])

        if b_out:
            win_agent = BLUE_OUT
            done = True
            # 出界需要修正最后的状态以避免状态超出界限RLLib报错
            # for i in range(4):
            #     self.state[1][i] = np.clip(self.state[1][i], self.low[i], self.high[i])
        return done, win_agent,ang_r_2_rb,ang_b_2_rb,dis

    def compuyte_one_step_reward_state(self, state_gym, red_action, blue_action, state_b):
        """
        返回 红蓝动作（仅仅组合成列表），基于red_action的下一步的状态s'，基于s'和给出的状态b 得出的奖励r'
        :param state_gym:
        :param red_action:
        :param blue_action:
        :param state_b:
        :return:
        """
        all_action = [red_action, blue_action]
        next_state_r = next_state(state_gym[0], red_action)[0]
        reward = self.reward_instance.re_init(state_gym[0], state_gym[1], next_state_r, state_b).get_reward()
        return all_action, next_state_r, reward

    def step_ana(self, state_gym, timestep, b_action_final):
        # global  total_time_steps
        reward_minmax_sum_1 = 0
        reward_model_sum_2 = 0

        state_b = next_state(state_gym[1], b_action_final)[0]

        r_action_1 = self.red_model1.get_action_from_td3_model(state_gym)
        r_action_2 = self.red_model2.get_action_from_td3_model(state_gym)
        # print(r_action_1,r_action_2)
        red_action_choices = [r_action_1, r_action_2]

        # if self.Red_
        # Model_Choice is not None:
        #     return red_action_choices[self.Red_Model_Choice]

        action_1, state_r_minmax, reward_minmax = self.compuyte_one_step_reward_state(state_gym, r_action_1,
                                                                                      b_action_final, state_b)
        action_2, state_r_model, reward_model = self.compuyte_one_step_reward_state(state_gym, r_action_2,
                                                                                    b_action_final, state_b)
        # print(reward_minmax)
        self.log.worksheet.write(total_time_steps + timestep, 4, reward_minmax)
        self.log.worksheet.write(total_time_steps + timestep, 5, reward_model)
        if self.FirstStep:
            # if self.GAMMA == 1:
            self.FirstStep = False

            reward_minmax_sum_1 ,ang_r_1_rb,ang_b_1_rb,dis_1= self.compute_multi_step_reward_minmax(reward_minmax, state_b,
                                                                      b_action_final, r_action_1, state_gym,
                                                                      action_1, state_r_minmax)
            reward_model_sum_2 ,ang_r_2_rb,ang_b_2_rb,dis_2= self.compute_multi_step_reward_model(reward_model, state_b,
                                                                    b_action_final, r_action_2, state_gym, action_2,
                                                                    state_r_model)

            # self.Red_Model_Choice = 1 if reward_minmax_sum_1 < reward_model_sum_2 else 0
            # if ang_r_1_rb and ang_r_2_rb != None:
            if dis_1 > dis_2 and ang_r_1_rb > ang_r_2_rb:
                self.Red_Model_Choice = 0
            else:
                self.Red_Model_Choice = 1
                # if ang_r_1_rb > ang_r_2_rb:
                #     self.Red_Model_Choice = 0
                # else:
                #     self.Red_Model_Choice = 1


                # if ang_r_1_rb - ang_b_1_rb   >   ang_r_2_rb - ang_b_2_rb:

            # else:
            #     self.Red_Model_Choice = 1 if reward_minmax_sum_1 < reward_model_sum_2 else 0
            # print(dis_1,dis_2)
            # print(ang_r_1_rb , ang_r_2_rb)
            # print(reward_minmax_sum_1 , reward_model_sum_2)
        # print('red',self.Red_Model_Choice)

        self.log.worksheet.write(total_time_steps + timestep, 6, reward_minmax_sum_1)
        self.log.worksheet.write(total_time_steps + timestep, 7, reward_model_sum_2)

        r_action = red_action_choices[self.Red_Model_Choice]
        self.log.worksheet.write(total_time_steps + timestep, 1, 'model vs diff' if self.Red_Model_Choice == 0 else 'model_e')
        # if reward_minmax_sum < reward_model_sum:
        # if :
        #     r_action = r_action_model
        #     self.log.worksheet.write(timestep, 1, 'model_e vs  diff')
        # else:
        #     r_action = r_action_minmax
        #     self.log.worksheet.write(timestep, 1, 'model4218')
        return r_action

    def compute_multi_step_reward_model(self, reward_model, state_b, b_action_final,
                                        r_action_model, state_gym, action_2, state_r_model):
        reward_model_sum = reward_model
        state_b_ = state_b.copy()
        b_action_ = list_one_dim_copy(b_action_final)
        r_action_ = list_one_dim_copy(r_action_model)

        state_model = copy.deepcopy(state_gym)

        for i in range(N_STEPS - 1):
            state_model[0] = state_r_model
            state_model[1] = state_b_
            action_2[0] = r_action_
            action_2[1] = b_action_
            state2, reward_model, done,ang_r_2_rb,ang_b_2_rb,dis = self.step_mulit(action_2, state_model)
            # reward_model
            # print('reward_model',reward_model)
            print('ang_r_2_rb,ang_b_2_rb',ang_r_2_rb,ang_b_2_rb)
            reward_model_sum += reward_model * (self.GAMMA ** i)
            if done:
                break
            r_action_ = self.red_model2.get_action_from_td3_model(state2)
            b_action_, R_a = get_blue_action_from_min_max_s(state2[1], state2[0])
            state_r_model = next_state(state_model[0], r_action_)[0]
            state_b_ = next_state(state_model[1], b_action_)[0]

        # state_model[0] = state_r_model
        # state_model[1] = state_b_
        # action_2[0] = r_action_
        # action_2[1] = b_action_
        # state2, reward_model, done, ang_r_2_rb, ang_b_2_rb, dis = self.step_mulit(action_2, state_model)
        # # reward_model
        # # print('reward_model',reward_model)
        # print('ang_r_2_rb,ang_b_2_rb', ang_r_2_rb, ang_b_2_rb)
        # if done:
        #     pass
        # else:
        #     r_action_ = self.red_model2.get_action_from_td3_model(state2)
        #     b_action_, R_a = get_blue_action_from_min_max_s(state2[1], state2[0])
        #     state_r_model = next_state(state_model[0], r_action_)[0]
        #     state_b_ = next_state(state_model[1], b_action_)[0]
        return reward_model_sum,ang_r_2_rb,ang_b_2_rb,dis

    def compute_multi_step_reward_minmax(self, reward_minmax, state_b,
                                         b_action_final, r_action_minmax, state_gym, action_1, state_r_minmax):
        reward_minmax_sum = reward_minmax
        state_b_ = state_b.copy()
        b_action_ = list_one_dim_copy(b_action_final)
        r_action_ = list_one_dim_copy(r_action_minmax)

        state_minmax = copy.deepcopy(state_gym)

        for i in range(N_STEPS - 1):
            state_minmax[0] = state_r_minmax
            state_minmax[1] = state_b_
            action_1[0] = r_action_
            action_1[1] = b_action_
            # print(type(env.range_5(action_1, state_gym)))
            state1, reward1_1, done ,ang_r_1_rb, ang_b_1_rb, dis = self.step_mulit(action_1, state_minmax)
            print('ang_r_1_rb, ang_b_1_rb, dis',ang_r_1_rb, ang_b_1_rb, dis)
            # print('reward_e',reward1_1)
            reward_minmax_sum += reward1_1 * (self.GAMMA ** i)
            if done:
                break
            r_action_, _ = get_blue_action_from_min_max_s(state1[0], state1[1])
            b_action_, R_a = get_blue_action_from_min_max_s(state1[1], state1[0])
            state_r_minmax = next_state(state_minmax[0], r_action_)[0]
            state_b_ = next_state(state_minmax[1], b_action_)[0]

        # state_minmax[0] = state_r_minmax
        # state_minmax[1] = state_b_
        # action_1[0] = r_action_
        # action_1[1] = b_action_
        # # print(type(env.range_5(action_1, state_gym)))
        # state1, reward1_1, done, ang_r_1_rb, ang_b_1_rb, dis = self.step_mulit(action_1, state_minmax)
        # print('ang_r_1_rb, ang_b_1_rb, dis', ang_r_1_rb, ang_b_1_rb, dis)
        # # print('reward_e',reward1_1)
        # if done:
        #     pass
        # else:
        #     r_action_, _ = get_blue_action_from_min_max_s(state1[0], state1[1])
        #     b_action_, R_a = get_blue_action_from_min_max_s(state1[1], state1[0])
        #     state_r_minmax = next_state(state_minmax[0], r_action_)[0]
        #     state_b_ = next_state(state_minmax[1], b_action_)[0]
        return reward_minmax_sum, ang_r_1_rb, ang_b_1_rb, dis


def get_blue_action(timestep, state_gym, log, blue_model, iblue):

    b_action_final = None
    if iblue == 1:

        b_action1 = blue_model.get_action_from_td3_model(state_gym)
        b_action_final = b_action1
        # b_action_final = None
        log.worksheet.write(total_time_steps + timestep, 8, 'model_diff')
    elif iblue == 2:
        b_action2, R_a = get_blue_action_from_min_max_s(state_gym[0], state_gym[1])
        b_action_final = b_action2
        log.worksheet.write(total_time_steps + timestep, 8, 'minmax')
    return b_action_final

def get_red_action(timestep, state_gym, log, red_model1,red_model2, ired):

    r_action_final = None
    if ired == 1:
        r_action_1 = red_model1.get_action_from_td3_model(state_gym)
        r_action_final = r_action_1
        log.worksheet.write(total_time_steps + timestep, 1, 'model vs diff')
    elif ired == 2:
        r_action_2 = red_model2.get_action_from_td3_model(state_gym)
        r_action_final = r_action_2
        log.worksheet.write(total_time_steps + timestep, 1, 'model_e')


    return r_action_final

total_time_steps = 0

def main():
    ############## Hyperparameters ##############
    # env_name = "UavBattle"
    # # creating environment
    # env = gym.make(env_name)
    # env = UavBattleEnvContinuousAction()
    env = UavBattleEnvContinuousAction()
    # print(env.observation_space.shape[0])
    # state_dim = 12
    # action_dim = 7
    render = False

    # solved_reward = 230  # stop training if avg_reward > solved_reward
    # log_interval = 100  # print avg reward in the interval
    max_episodes = 300  # max training episodes
    max_timesteps = 300  # max timesteps in one episode
    #
    # n_latent_var = state_dim**2  # number of variables in hidden layer
    # update_timestep = 2000  # update policy every n timesteps
    # lr = 0.002
    # betas = (0.9, 0.999)  # 中心距，更新权重参数
    # gamma = 0.99  # discount factor折扣因子
    # K_epochs = 4  # update policy for K epochs
    # eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    # ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    # print(lr,betas)

    # logging variables
    # state_builder = ["minmax算法", get_blue_action_from_min_max, get_random_state]
    # running_reward = 0
    # total_time_steps = 0  # 跑的最大长度
    red_win_nums = 0

    # model2 是指数函数
    red_model1 = Model(os.path.abspath('./各代优势函数/DQN_UAV_Battle_2022-10-11_08-52-197y77mvfw/model'))
    # red_model1 = Model(os.path.abspath('./DQN_UAV_Battle_2022-04-10_15-06-00rbq89fvl/model'))
    # red_model2 = Model(os.path.abspath('./各代优势函数/DQN_UAV_Battle_2022-10-11_08-52-197y77mvfw/model'))
    red_model2 = Model(os.path.abspath('./DQN_UAV_Battle_2022-04-10_15-06-00rbq89fvl/model'))
    # red_model1 = Model(os.path.abspath('./model-e vs diff  98/model'))
    # red_model2 = Model(os.path.abspath('./model e vs  diff/model'))
    # blue_model = Model(os.path.abspath('./diff/model'))
    # blue_model = Model(os.path.abspath('./各代优势函数/DQN_UAV_Battle_2022-10-06_05-44-16iucjr_cq/model'))
    # blue_model = Model(os.path.abspath('./各代优势函数/DQN_UAV_Battle_2022-10-06_02-55-48zfezlqtb/model'))
    blue_model = None
    print(red_model2)
    print(red_model1)
    # training loop
    log = LogAna()  # 日志分析初始化
    global total_time_steps
    GAMMA = 0.99
    ana = Analyse(GAMMA, red_model1, red_model2, log)  # 红方动作分析类 初始化
    R_s = np.zeros((7, 7))  # 输出minmax矩阵
    R_e = np.zeros((7, 7))

    for i_episode in range(1, max_episodes + 1):  # 50000
        state_gym = np.array(copy.deepcopy(env.reset()))  # 初始化（重新玩）
        # iblue = random.randint(1, 2)
        iblue = random.randint(2, 2)
        ired = random.randint(1, 2)
        timestep = 0
        # ana.FirstStep = True
        ana.re_init()
        info = None
        for t in range(max_timesteps):  # 300，卡死循环的情况，不坠落
            timestep += 1
            # print(state_gym)
            # print(state_gym[::-1])
            # if t < 5:
            #     b_action_ = get_blue_action(timestep, copy.deepcopy(state_gym[::-1]), log, blue_model, iblue)
            #     r_action_ = get_red_action(timestep, copy.deepcopy(state_gym[::-1]), log, red_model1,red_model2, ired)
            #     state_, reward, done, info = env.step([r_action_, b_action_])
            #     state_gym = copy.deepcopy(state_)
            #     log.worksheet.write(total_time_steps + timestep, 2, reward)
            #     log.worksheet.write(total_time_steps + timestep, 0, t + 1)
            #     if render:
            #         env.render()
            #     if done:
            #         break
            # else:
            b_action_final = get_blue_action(timestep, copy.deepcopy(state_gym[::-1]), log, blue_model, iblue)
            # print(state_gym)
            # print(b_action_final)
            r_action_final = ana.step_ana(copy.deepcopy(state_gym), timestep, b_action_final)
            state_, reward, done, info = env.step([r_action_final, b_action_final])  # 得到（新的状态，奖励，是否终止，额外的调试信息）
            state_gym = copy.deepcopy(state_)

            log.worksheet.write(total_time_steps + timestep, 2, reward)
            log.worksheet.write(total_time_steps + timestep, 0, t + 1)
            if render:
                env.render()
            if done:
                break

        # print(info)
        win_agent = info['win_agent']

        total_time_steps += timestep
        win_agent_str = episode_ends_describe[win_agent]
        log.worksheet.write(total_time_steps , 3, win_agent_str)

        print('Episode {} \t timestep: {} \t  win_agent:{}'.format(i_episode, timestep, win_agent_str))
        # total_time_steps = 0
        if win_agent == RED_WIN:
            red_win_nums += 1
        if i_episode % 100 == 0:
            print("########## 胜率! ##########")
            print('红方胜率{}'.format(red_win_nums / 100))
            red_win_nums = 0

    # np.savetxt('outputprint_s.txt', R_s)
    # np.savetxt('outputprint_e.txt', R_e)

    log.workbook.close()

    # stop training if avg_reward > solved_reward
    # if win_agent == 1:
    #     b += 1
    #     if i_episode % 1000 == 0 and b / i_episode >= 0.9:
    #         print("########## Solved! ##########")
    #         torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format('uav_dis'))
    #         break

    # logging
    # if i_episode % 1 == 0:
    #     # avg_length = int(avg_length / log_interval)
    #     # running_reward = int((running_reward / log_interval))
    #     print('Episode {} \t avg length: {} \t  win_agent:{}'.format(i_episode, avg_length, win_agent))
    #     # print('Episode {} \t avg length: {} \t reward: {} \t win_agent:{}'.format(i_episode, avg_length, running_reward,win_agent))
    #     running_reward = 0
    #     avg_length = 0


def list_one_dim_copy(a):
    return [i for i in a]


if __name__ == '__main__':
    main()

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
