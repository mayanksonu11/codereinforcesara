#import gym
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self,state_space_size):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(state_space_size, 128)
        self.fc2 = nn.Linear(128, 3)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

# def main():
#     env = gym.make('CartPole-v1')
#     pi = Policy()
#     score = 0.0
#     print_interval = 20
    
    
#     for n_epi in range(10000):
#         s, _ = env.reset()
#         done = False
        
#         while not done: # CartPole-v1 forced to terminates at 500 step.
#             prob = pi(torch.from_numpy(s).float())
#             m = Categorical(prob)
#             a = m.sample()
#             s_prime, r, done, truncated, info = env.step(a.item())
#             pi.put_data((r,prob[a]))
#             s = s_prime
#             score += r
            
#         pi.train_net()
        
#         if n_epi%print_interval==0 and n_epi!=0:
#             print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
#             score = 0.0
#     env.close()

class Agent(object):
    """Deep Q-learning agent."""

    def __init__(self,
                 state_space_size,
                 action_space_size,
                 path="",
                 target_update_freq=100, #1000, #cada n steps se actualiza la target network
                 discount=0.99,
                 batch_size=32,
                 max_explore=1,
                 min_explore=0.05,
                 anneal_rate=(1/5000), #100000),
                 replay_memory_size=100000,
                 replay_start_size= 500): #500): #10000): #despues de n steps comienza el replay
        """Set parameters, initialize network."""
        self.action_space_size = action_space_size
        
        self.q = Policy(state_space_size)
        # self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate)
        if(path != ""):
            self.q.load_state_dict(torch.load(path))
            self.q.eval()
        self.q_target = Policy(action_space_size)
        self.q_target.load_state_dict(self.q.state_dict())
        self.score=0

        #self.online_network = Network(state_space_size, action_space_size)##########Qnet se change karna h
        #self.target_network = Network(state_space_size, action_space_size)#############

        # self.update_target_network()

        # training parameters
        self.target_update_freq = target_update_freq
        self.discount = discount
        self.batch_size = batch_size

        # policy during learning
        self.max_explore = max_explore + (anneal_rate * replay_start_size)
        self.min_explore = min_explore
        self.anneal_rate = anneal_rate
        self.steps = 0

        # # replay memory
        # self.memory = ReplayBuffer(replay_memory_size)
        # self.replay_start_size = replay_start_size
        # self.experience_replay = ReplayBuffer(replay_memory_size)

    def handle_episode_start(self):
        self.last_state, self.last_action = None, None

    def step(self, state, reward, training=True):###############step function me apis se change karne h and call fuction dekhna h
        
        """Observe state and rewards, select action.
        It is assumed that `observation` will be an object with
        a `state` vector and a `reward` float or integer. The reward
        corresponds to the action taken in the previous step.
        """

        last_state, last_action = self.last_state, self.last_action
        last_reward = reward
        state = state

        epsilon = max(0.01, 0.08 - 0.01*(self.steps/200)) #Linear annealing from 8% to 1%
        print_interval = 20
        print("Current State:",state)

        prob = self.q(torch.from_numpy(np.asarray(state)).float())
        self.m = Categorical(prob) 
        action = self.m.sample()      
        # if last_state!=None:
        #     self.memory.put((last_state,last_action,last_reward/100.0,state))
       
       
                   
       # if self.memory.size()>2000:
        self.q.put_data((reward,prob[action]))

        self.score += reward
        self.q.train_net()

        if self.steps%print_interval==0 and self.steps!=0:
            self.q_target.load_state_dict(self.q.state_dict())
            print("n_episode :{}, score : {:.1f}, eps : {:.1f}%".format(
                                                            self.steps, self.score/print_interval, epsilon*100))
            self.score = 0.0
        self.steps += 1

        self.last_state = state
        self.last_action = action

        return action

    # def policy(self, state, training):
    #     """Epsilon-greedy policy for training, greedy policy otherwise."""
    #     explore_prob = self.max_explore - (self.steps * self.anneal_rate)#probabilidad de exploracion decreciente
    #     explore = max(explore_prob, self.min_explore) > np.random.rand()

    #     if training and explore: #hacer exploracion
    #         action = np.random.randint(self.action_space_size)
    #     else: #hacer explotacion
    #         inputs = np.expand_dims(state, 0)
    #         qvalues = self.online_network.model(inputs) #online or evalation network predicts q-values
    #         #print("***##qvalues",qvalues)
    #         action = np.squeeze(np.argmax(qvalues, axis=-1))#need work

    #     return action
    def save(self):
        torch.save(self.q.state_dict(),"./reinforce.pt")

    
# if __name__ == '__main__':
#     main()