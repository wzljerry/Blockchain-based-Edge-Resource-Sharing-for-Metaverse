# -*- coding: utf-8 -*-
'''
@Title : Blockchain-based Edge Resource Sharing for Metaverse
@Author : Zhilin Wang
@Email : wangzhil@iu.edu
@Date : 14-04-2022
'''
import numpy as np
import math
import random
random.seed(1)
import data_collection as dc
import task_allocation as ta

if __name__=='__main__':
  #parameters
  num_server=20#number of servers
  num_task=50#number of tasks
  EPSILON=0.9 # greedy
  GAMMA=0.9 #discount
  ALPHA=0.01 #learning rate
  max_iteration=500#iterations
  #generate data
  data=dc.DataCollection()
  p=data.generate_random_value(1,10,num_task) #price
  D=data.generate_random_value(10,20,num_task) #data size
  T=data.generate_random_value(1,100,num_task) #time
  task=data.generate_task(p,D,T,num_task) #tasks
  task=np.array(task)
  ser_task=data.generate_group(num_task,num_server) #server-class group
  server=data.generate_server(200,400,num_server) #server
  f_j=data.generate_random_value(1,10,num_server) #CPU frequency
  d_i=data.generate_random_value(1,10,num_server)
  alpha=0.00001
  B=data.generate_random_value(5,10,num_server)
  P_i=data.generate_random_value(5,10,num_server)
  G_i=data.generate_random_value(5,10,num_server)
  delta=0.001
  R_i_j=[]
  for i in range(num_server):
    R=B[i]*math.log(1+(P_i[i]*G_i[i])/delta**2)
    R_i_j.append(R)

  #training
  #self,f_j, d_i, alpha, B, P_i, G_i, task,server,ser_task,ALPHA,GAMMA,EPSILON,MAX_EPISODES
  ql=ta.Task_allocation(f_j, d_i, alpha, B, P_i, G_i, R_i_j,delta,task,server,ser_task,EPSILON,ALPHA,GAMMA,max_iteration)
  #random
  max_random=ql.ramdom_select()
  print(max_random)
  #greedy
  max_greedy=ql.greedy_select()
  print(max_greedy)
  #q_learning
  max_reward,solution,q_table,res=ql.training()
  solution=np.array(solution)
  #print(solution)
  print(max_reward)
  print(res)
  #print(q_table)