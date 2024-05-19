import numpy as np
import math
import random
from gym.utils import seeding

random_env_low = np.array([3500, 3500, 750, 50, -math.pi, -math.pi])
random_env_high = np.array([6500, 6500, 3250, 120, math.pi, math.pi])
attack_dis = 1000
radar_dis = 2500
weight_d = 0.3
weight_a = 0.3
weight_v = 0.4
weight_t_type = 0.2
weight_t_num = 0.2
weight_t_s1 = 0.6


class Threat:
    """
     Observation:
        状态空间勇敢两个6维的数组表示 分别表示对抗的两个无人机飞行状态
        每个数组包括 东北天坐标系下的三维空间未知 x y z，速度 v，俯仰角theta，偏航角psi
        Type：Tuple(Box(6),Box(6))
        Num       Observation       Min       Max
        0           x               0         10000
        1           y               0         10000
        2           z               200       4000
        3           v               50        120
        4           theta           -inf      inf
        5           psi             -inf      inf
    """
    def __init__(self,s_red,s_blue):
        self.s_red = s_red
        self.s_blue = s_blue

    def i_to_f(self,x):
        if x == 1:
            x = 1
        elif x == 2:
            x = 0.875
        elif x == 3:
            x = 0.75
        elif x == 4:
            x = 0.625
        elif x == 5:
            x = 0.5
        elif x == 6:
            x = 0.375
        elif x == 7:
            x = 0.25
        elif x == 8:
            x = 0.1
        return x

    def f_to_i(self,x):
        if 0.125 > x >= 0:
            x = 8
        elif 0.25 > x >= 0.125:
            x = 7
        elif 0.375 > x >= 0.25:
            x = 6
        elif 0.5 > x >= 0.375:
            x = 5
        elif 0.625 > x >= 0.5:
            x = 4
        elif 0.75 > x >= 0.625:
            x = 3
        elif 0.875 > x >= 0.75:
            x = 2
        elif 1 > x >= 0.875:
            x = 1
        return x

    def threat_assessment(self,t):
        s1 = self.s1_offensive_advantage()
        s1 = self.i_to_f(s1)
        if t[2] == 0:
            z_t = weight_t_type * t[0] + weight_t_num * t[1] + weight_t_s1 * s1
        else:
            z_t = 1
        return z_t

    def s1_offensive_advantage(self):
        s1_t_d = self.t_distance(self.s_red, self.s_blue)
        s1_t_a = self.t_angle(self.s_red, self.s_blue)
        s1_t_v = self.t_velocity(self.s_red, self.s_blue)

        s1 = weight_d * s1_t_d + weight_a * s1_t_a + weight_v * s1_t_v
        s1 = self.f_to_i(s1)
        return s1


    def t_distance(self,s_red,s_blue):
        dis = np.linalg.norm(
            np.array([s_red[0] - s_blue[0], s_red[1] - s_blue[1], s_red[2] - s_blue[2]]))
        if dis > radar_dis:
            t_d = 0
        elif dis < attack_dis:
            t_d = 1
        else:
            t_d = 1 - (dis - attack_dis)/(radar_dis - attack_dis)
        return t_d


    def t_angle(self,s_red,s_blue):
        vector_red_2_blue = np.array(
            [s_blue[0] - s_red[0], s_blue[1] - s_red[1], s_blue[2] - s_red[2]])  # 以红方为起点到蓝方的向量
        vector_v_red = np.array(
            [[s_red[3] * math.cos(s_red[4]) * math.cos(s_red[5])], [s_red[3] * math.cos(s_red[4]) * math.sin(s_red[5])],
             [s_red[3] * math.sin(s_red[4])]])  # red x_dot, y_dot, z_dot

        vector_v_blue = np.array(
            [[s_blue[3] * math.cos(s_blue[4]) * math.cos(s_blue[5])],
             [s_blue[3] * math.cos(s_blue[4]) * math.sin(s_blue[5])],
             [s_blue[3] * math.sin(s_blue[4])]])  # red x_dot, y_dot, z_dot

        ang_r_2_rb = math.acos(
            np.dot(vector_red_2_blue, vector_v_red) / (np.linalg.norm(vector_red_2_blue) * np.linalg.norm(
                vector_v_red)))  # 红方速度与红蓝向量的夹角 phi

        ang_b_2_rb = math.acos(
            np.dot(vector_red_2_blue, vector_v_blue) / (np.linalg.norm(vector_red_2_blue) * np.linalg.norm(
                vector_v_blue)))  # 蓝方速度与红蓝向量的夹角  theta
        # t_a = (abs(ang_r_2_rb) - abs(ang_b_2_rb) + math.pi) / (2 * math.pi)
        t_a = (ang_r_2_rb + ang_b_2_rb) / (2 * math.pi)
        return t_a

    def t_velocity(self,s_red,s_blue):
        if s_red[3] < 0.6 * s_blue[3]:
            t_v = 0.1
        elif 0.6 * s_blue[3] <= s_red[3] <= 1.5 * s_blue[3]:
            t_v = -0.5 + s_red[3] / s_blue[3]
        else:
            t_v = 1
        return t_v

def main():
    np_random, seed = seeding.np_random(None)
    data = []
    with open('data.txt', mode='w') as fp:
        for i in range(18000):
            observation_red = [np_random.uniform(random_env_low[i], random_env_high[i]) for i in
                               range(len(random_env_low))]
            observation_blue = [np_random.uniform(random_env_low[i], random_env_high[i]) for i in
                                range(len(random_env_low))]
            threat = Threat(observation_red, observation_blue)

            t_type = random.randint(1, 8)      # t1
            t_type = threat.i_to_f(t_type)
            t_num = random.randint(1, 8)       # t2
            t_num = threat.i_to_f(t_num)
            t_mw = random.randint(0, 1)        # t3 missile warning
            s_1 = threat.s1_offensive_advantage()  # s1
            s_1 = threat.i_to_f(s_1)
            t = [t_type, t_num, t_mw, s_1]
            z = round(threat.threat_assessment(t), 3)
            z = threat.f_to_i(z)
            z = z-1
            # z = random.randint(0, 7)
            t.append(z)
            data.append(t)
        np.savetxt(fp, data, fmt='%f', delimiter=',')
            # fp.writelines(str(data)+ '\n')
    fp.close()


if __name__ == '__main__':
    main()
    print('hello')


