# Blockchain-based-Edge-Resource-Sharing-for-Metaverse
This paper has been accepted to IEEE MASS 2022. [Link](https://arxiv.org/abs/2208.05120)

# How to run this project?

1. We can run both python or ipynb files directly after setting the parameter.
2. We can run main.py in MTA_QL directly.

# Problem Description
We are going to solve a multiple task allocation problem on mobile edge servers. The system has m tasks and n servers, and m>n. Each task has it's price  ,data size, and time limitation for computing; and each server has its computing capacity, CPU architecture, and communication bandwidth. Our goal is the find a best allocation scheme to assign each task to one server and get the maximum payments for the whole system. 

This is a mutiple knapsack problem (MKP), which is NP-hard. 


# Details
There are three solutions provided. 

### Solution based on Q_learning.
We let each task as the state space, and each server as the action. The reason is that we would like to make sure that each task can be only assigned to one server. We get a m*n matrix.

We collect the reward table for (m,n), and we select the avaliable actions for each state by considering the CPU and time constraints.  We also need to keep a Q-table which will be applied to select action for each state. Then we apply epsilon-greedy to select action. With multiple rounds of training, we can have the best solution.


### Solution based on greedy search.
Each state selects the maximum reward from avaliable actions.



### Solution based on random search.
Each state selects the actions randomly.


### Please cite:

@inproceedings{wang2022blockchain,
  title={Blockchain-based Edge Resource Sharing for Metaverse},
  author={Wang, Zhilin and Hut, Qin and Xu, Minghui and Jiang, Honglu},
  booktitle={2022 IEEE 19th International Conference on Mobile Ad Hoc and Smart Systems (MASS)},
  pages={620--626},
  year={2022},
  organization={IEEE}
}
