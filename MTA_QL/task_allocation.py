import numpy as np
import pandas as pd
import random
from random import choice
#random.seed(1)


class Task_allocation:
    # parameters
    # task is a matrix which contains [price p_j, data size D_j, executing time t_e]
    # server is a list, which contains the computational powers for each server
    def __init__(self, f_j, d_i, alpha, B, P_i, G_i, R_i_j, delta, task, server, ser_task, ALPHA, GAMMA, EPSILON,
                 MAX_EPISODES):
        self.task = task
        self.lamda = 0.01
        self.ser_task = ser_task
        self.server = server
        self.m = task.shape[0]  # number of tasks
        self.n = len(server)  # number of servers
        self.B = B  # bandwidth
        self.delta = 0.001  # gussian noise
        self.alpha = alpha  # CPU parameter
        self.P_i = P_i  # trainsimission power
        self.G_i = G_i  # trainsimission gain
        self.f_j = f_j  # CPU frequency
        self.d_i = d_i  # sample cpu cyecles
        self.R_i_j = R_i_j
        self.N_STATES = self.m  # number of states
        self.ACTIONS = [i for i in range(self.n)]  # action
        self.EPSILON = EPSILON  # greedy
        self.ALPHA = ALPHA  # learning rate
        self.GAMMA = GAMMA  # discount
        self.MAX_EPISODES = MAX_EPISODES
        self.q_table = self.build_q_table()  # q_table

    # generate values which can be repeated
    def generate_random_value(self, low, up, num):
        randomlist = []
        for i in range(num):
            n = random.randint(low, up)
            randomlist.append(n)
        return randomlist

        # create the q_table with initial value=0

    def build_q_table(self):
        q_table = pd.DataFrame(
            np.zeros((self.N_STATES, len(self.ACTIONS))),  # initialize q_table
            columns=self.ACTIONS,  # columns, the name of actions
        )
        return q_table

    # check whether the server i can process task j with cpu and time constraints
    def check_avaliable(self, j, i, h):
        res = 0
        if self.task[j][1] * self.d_i[i] < self.server[i]:  # cpu
            if (self.task[j] * self.d_i[i] / self.f_j[i] + self.task[j][1] / self.R_i_j[h]).all() < self.task[j][
                2]:  # time
                res = 1
        return res

    # if j is from i, then no transimision
    def check_avaliable_1(self, j, i):
        res = 0
        if self.task[j][1] * self.d_i[i] < self.server[i]:  # cpu
            if (self.task[j] * self.d_i[i] / self.f_j[i]).all() < self.task[j][2]:  # time
                res = 1
        return res

    def find_source(self, j):
        for com in self.ser_task:
            if j in com:
                index = com.index(j)
        return index

    # check whether task j is from server i
    def check_source(self, task_num, group_num):
        group = self.ser_task[group_num]  # group
        res = 0
        if task_num in group:  # in or not
            res = 1
        return res

    # get the reward based on sources and avaliability
    def get_feedback(self):
        rew = []
        for j in range(self.m):  # for each task
            for i in range(self.n):  # for each server
                if self.check_source(j, i) == 1:  # in that group
                    # h=i
                    if self.check_avaliable_1(j, i) == 1:  # avaliable
                        reward = self.task[j][0] * self.task[j][1] * self.d_i[i]
                        -self.alpha * self.task[j][1] * self.d_i[i] * self.f_j[i] ** 2  # only need to compute
                    else:
                        reward = -self.task[j][1] / self.R_i_j[i] * self.P_i[i] + self.lamda * self.task[j][0] * \
                                 self.task[j][1] * self.d_i[i]  # only need to transimist
                if self.check_source(j, i) != 1:  # not in that group
                    h = self.find_source(j)
                    if self.check_avaliable(j, i, h) == 1:  # avaliable
                        reward = self.task[j][0] * self.task[j][1] * self.d_i[i]
                        -self.alpha * self.task[j][1] * self.d_i[i] * self.f_j[i] ** 2 - self.lamda * self.task[j][0] * \
                        self.task[j][1] * self.d_i[i]  # needs to compute
                    else:
                        reward = 0  # get nothing because it can't process task j
                rew.append(reward)  # using a list to contain all the rewards for each task and each server
        return rew

    # to get the avaliable actions set for each task
    def check_action_reward(self, task_num):
        reward = self.get_feedback()
        reward = np.array(reward).reshape(self.m, self.n)  # using a m*n matrix to contain the rewards
        index_list = []
        for i in range(self.n):  # for each server
            if reward[task_num][i] != 0:  # if =0, the server can't process that task
                action_index = i
                index_list.append(action_index)
        return index_list  # return the avaliable index of actions

        # to get the avaliable actions set for each task without speed up

    def check_action_reward_without_speed_up(self, task_num):
        reward = self.get_feedback()
        reward = np.array(reward).reshape(self.m, self.n)  # using a m*n matrix to contain the rewards
        index_list = []
        for i in range(self.n):  # for each server
            # if reward[task_num][i]!=0: # if =0, the server can't process that task
            action_index = i
            index_list.append(action_index)
        return index_list  # return the avaliable index of actions

    def select_action(self, task_num, state_actions):
        # state_actions = self.check_action_reward(task_num)
        if (np.random.uniform() > self.EPSILON) or (len(state_actions) == 0):
            # if np.random.uniform() > self.EPSILON:  # not greedy
            # action_name = np.random.choice(state_actions)
            action_name = choice(state_actions)

        else:
            q_value = pd.DataFrame(self.q_table.loc[task_num, state_actions]).T
            action_name = int(q_value.idxmax(axis=1))  # greedy
        return action_name

    # create a table to contain the cpu capacity
    def cpu_table(self):
        cpu_table = self.build_q_table()
        return cpu_table

    # create a table to contain the time
    def time_table(self):
        time_table = self.build_q_table()
        return time_table

    def check_action_valid(self, j, actions_ava, rewards, cpu_table, time_table, state_actions, action, acc_mu, acc_t):
        res = True
        if action in actions_ava:  # select actions in avaliable sets
            if (acc_mu > self.server[action] or acc_t < self.task[j][
                2]):  # satisify that the cpu and time are both enough
                state_actions.remove(action)  # remove that action
                action = self.select_action(j, state_actions)
                if (acc_mu > self.server[action] or acc_t < self.task[j][2]):
                    action = self.check_action_valid(j, actions_ava, rewards, cpu_table, time_table, state_actions,
                                                     action, acc_mu, acc_t)
            else:
                actions_ava.append(action)
        return action

    def q_update_modified(self):
        actions_ava = []
        rews = []
        act = []
        rewards = np.array(self.get_feedback()).reshape(self.m, self.n)
        cpu_table = self.cpu_table()
        time_table = self.time_table()
        # check the limitation of time and cpu
        for j in range(self.m):
            state_actions = self.check_action_reward(j)
            action = self.select_action(j, state_actions)
            # initialize acc_t and acc_mu
            acc_mu = cpu_table[action].sum()  # sum of cpu
            acc_t = (self.server[action] - acc_mu) / self.f_j[action]
            action = self.check_action_valid(j, actions_ava, rewards, cpu_table, time_table, state_actions, action,
                                             acc_mu, acc_t)

            # record the cpu and time
            cpu_table[action][j] = self.task[j][1]
            acc_mu = cpu_table[action].sum()  # sum of cpu

            # update q
            if j != self.m - 1:
                self.q_table.iloc[j, action] += self.ALPHA * (rewards[j][action]
                                                              + self.GAMMA * np.max(
                            self.q_table.iloc[j + 1, state_actions]) - self.q_table.iloc[j, action])
            else:
                self.q_table.iloc[j, action] += self.ALPHA * (rewards[j][action]
                                                              + self.GAMMA * np.max(
                            self.q_table.iloc[j, state_actions]) - self.q_table.iloc[j, action])
            rew = rewards[j][action]  # the rewards of each step
            rews.append(rew)  # rewads for all the tasks
            act.append(action)  # action set
        res = sum(rews)  # total rewards for one tempt
        return self.q_table, res, act

    # update q_table
    def q_update(self):
        actions_ava = []
        rews = []
        act = []
        rewards = np.array(self.get_feedback()).reshape(self.m, self.n)
        cpu_table = self.cpu_table()
        time_table = self.time_table()
        # check the limitation of time and cpu
        for j in range(self.m):
            state_actions = self.check_action_reward(j)
            action = self.select_action(j, state_actions)
            # initialize acc_t and acc_mu
            acc_mu = cpu_table[action].sum()  # sum of cpu
            acc_t = (self.server[action] - acc_mu) / self.f_j[action]
            if action in actions_ava:  # select actions in avaliable sets
                if (acc_mu > self.server[action] or acc_t < self.task[j][
                    2]):  # satisify that the cpu and time are both enough
                    state_actions.remove(action)  # remove that action
                    action = self.select_action(j, state_actions)  # reselect another action
            else:
                actions_ava.append(action)

            # record the cpu and time
            cpu_table[action][j] = self.task[j][1]
            acc_mu = cpu_table[action].sum()  # sum of cpu

            # update q
            if j != self.m - 1:
                self.q_table.iloc[j, action] += self.ALPHA * (rewards[j][action]
                                                              + self.GAMMA * np.max(
                            self.q_table.iloc[j + 1, state_actions]) - self.q_table.iloc[j, action])
            else:
                self.q_table.iloc[j, action] += self.ALPHA * (rewards[j][action]
                                                              + self.GAMMA * np.max(
                            self.q_table.iloc[j, state_actions]) - self.q_table.iloc[j, action])
            rew = rewards[j][action]  # the rewards of each step
            rews.append(rew)  # rewads for all the tasks
            act.append(action)  # action set
        res = sum(rews)  # total rewards for one tempt
        return self.q_table, res, act

        # update q_table

    def q_update_without_speed_up(self):
        actions_ava = []
        rews = []
        act = []
        rewards = np.array(self.get_feedback()).reshape(self.m, self.n)
        cpu_table = self.cpu_table()
        time_table = self.time_table()
        # check the limitation of time and cpu
        for j in range(self.m):
            state_actions = self.check_action_reward_without_speed_up(j)
            action = self.select_action(j, state_actions)
            # initialize acc_t and acc_mu
            acc_mu = cpu_table[action].sum()  # sum of cpu
            acc_t = (self.server[action] - acc_mu) / self.f_j[action]
            action = self.check_action_valid(j, actions_ava, rewards, cpu_table, time_table, state_actions, action,
                                             acc_mu, acc_t)
            # record the cpu and time
            cpu_table[action][j] = self.task[j][1]
            acc_mu = cpu_table[action].sum()  # sum of cpu

            # update q
            if j != self.m - 1:
                self.q_table.iloc[j, action] += self.ALPHA * (rewards[j][action]
                                                              + self.GAMMA * np.max(
                            self.q_table.iloc[j + 1, state_actions]) - self.q_table.iloc[j, action])
            else:
                self.q_table.iloc[j, action] += self.ALPHA * (rewards[j][action]
                                                              + self.GAMMA * np.max(
                            self.q_table.iloc[j, state_actions]) - self.q_table.iloc[j, action])
            rew = rewards[j][action]  # the rewards of each step
            rews.append(rew)  # rewads for all the tasks
            act.append(action)  # action set
        res = sum(rews)  # total rewards for one tempt
        return self.q_table, res, act

    # get the index based on greedy
    def greedy_reward(self, j, rewards):
        rewards = pd.DataFrame(rewards)
        index = self.check_action_reward(j)
        reward = rewards.iloc[j, index]
        max_reward = max(list(reward))
        max_index = list(reward).index(max_reward)
        return max_index

    # get the rewards by greedy search
    def greedy_select(self):
        actions_ava = []
        rews = []
        act = []
        rewards = np.array(self.get_feedback()).reshape(self.m, self.n)
        cpu_table = self.cpu_table()
        time_table = self.time_table()
        # check the limitation of time and cpu
        for j in range(self.m):
            state_actions = self.check_action_reward(j)
            action = self.greedy_reward(j, rewards)
            # initialize acc_t and acc_mu
            acc_mu = cpu_table[action].sum()  # sum of cpu
            acc_t = (self.server[action] - acc_mu) / self.f_j[action]
            if action in actions_ava:  # select actions in avaliable sets
                if (acc_mu > self.server[action] or acc_t < self.task[j][
                    2]):  # satisify that the cpu and time are both enough
                    state_actions.remove(action)  # remove that action
                    action = self.select_action(j, state_actions)  # reselect another action
            else:
                actions_ava.append(action)

            # record the cpu and time
            cpu_table[action][j] = self.task[j][1]
            acc_mu = cpu_table[action].sum()  # sum of cpu

            res = rewards[j][action]
            rews.append(res)
        rews_sum = sum(rews)
        return rews_sum

    # get the index randomly selected
    def random_index(self, j, rewards):
        rewards = pd.DataFrame(rewards)
        # for j in range(self.m):
        index = self.check_action_reward(j)
        reward = rewards.iloc[j, index]
        random_reward = choice(list(reward))
        random_index = list(reward).index(random_reward)
        return random_index

    # get the rewards by random strategy
    def ramdom_select(self):
        actions_ava = []
        rews = []
        act = []
        rewards = np.array(self.get_feedback()).reshape(self.m, self.n)
        cpu_table = self.cpu_table()
        time_table = self.time_table()
        # check the limitation of time and cpu
        for j in range(self.m):
            state_actions = self.check_action_reward(j)
            action = self.random_index(j, rewards)
            # initialize acc_t and acc_mu
            acc_mu = cpu_table[action].sum()  # sum of cpu
            acc_t = (self.server[action] - acc_mu) / self.f_j[action]

            if action in actions_ava:  # select actions in avaliable sets
                if (acc_mu > self.server[action] or acc_t < self.task[j][
                    2]):  # satisify that the cpu and time are both enough
                    state_actions.remove(action)  # remove that action
                    action = self.select_action(j, state_actions)  # reselect another action
            else:
                actions_ava.append(action)

            # record the cpu and time
            cpu_table[action][j] = self.task[j][1]
            acc_mu = cpu_table[action].sum()  # sum of cpu

            res = rewards[j][action]
            rews.append(res)
        rews_sum = sum(rews)
        return rews_sum

    # training with speed up
    def training(self):
        res = []
        act = []
        # training
        for i in range(self.MAX_EPISODES):
            q_table, reward, actions = self.q_update_modified()
            res.append(reward)
            act.append(actions)
        max_reward = np.max(res)
        best_solution = act[res.index(np.max(res))]
        index_list = [j for j in range(self.m)]
        best_server_task = list(zip(index_list, best_solution))  # the best allocation
        return max_reward, best_server_task, q_table, res

    # training without speed up
    def training_without_speed_up(self):
        res = []
        act = []
        # training
        for i in range(self.MAX_EPISODES):
            q_table, reward, actions = self.q_update_without_speed_up()
            res.append(reward)
            act.append(actions)
        max_reward = np.max(res)
        best_solution = act[res.index(np.max(res))]
        index_list = [j for j in range(self.m)]
        best_server_task = list(zip(index_list, best_solution))  # the best allocation
        return max_reward, best_server_task, q_table, res