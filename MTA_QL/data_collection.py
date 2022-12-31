import numpy as np
import random
from random import shuffle
#random.seed(1)


# generate data
class DataCollection:
    def __init__(self):
        pass

    # get random value
    def generate_random_value(self, low, up, num):
        randomlist = []
        for i in range(num):
            n = random.randint(low, up)
            randomlist.append(n)
        return randomlist

    # generate task
    def generate_task(self, p, D, T, num):
        task = []
        for i in range(num):
            a = [p[i], D[i], T[i]]
            task.append(a)
        return task

    # generate server
    def generate_server(self, low, up, num):
        server = self.generate_random_value(low, up, num)
        return server

    # generate group
    def generate_group(self, num_task, num_server):
        num = [i for i in range(num_task)]
        shuffle(num)
        group_task = np.array_split(num, num_server)
        for i in range(num_server):
            group_task[i] = list(group_task[i])
        return group_task