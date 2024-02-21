#import gym
import collections
import random
import numpy as np
# import tensorflow as tf
import collections as cns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.90
#buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self,max_size):
        self.buffer = collections.deque(maxlen=max_size)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst= [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float)
               
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 30)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            x= random.randint(0,29)
            print(x)
            return x
            #return 1 
        else :
            print(1) 
            return out.argmax()
    
   


            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class Agent(object):
    """Deep Q-learning agent."""

    def __init__(self,
                 state_space_size,
                 action_space_size,
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

        self.q = Qnet()
        self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate)
        self.q_target = Qnet()
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

        # replay memory
        self.memory = ReplayBuffer(replay_memory_size)
        self.replay_start_size = replay_start_size
        self.experience_replay = ReplayBuffer(replay_memory_size)

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
        print(state)

        action = self.q.sample_action(torch.from_numpy(np.array(state)).float(), epsilon)      
        if last_state!=None:
            self.memory.put((last_state,last_action,last_reward/100.0,state))
       
        self.score += reward
                   
        if self.memory.size()>2000:
            train(self.q, self.q_target, self.memory, self.optimizer)

        if self.steps%print_interval==0 and self.steps!=0:
            self.q_target.load_state_dict(self.q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            self.steps, self.score/print_interval, self.memory.size(), epsilon*100))
            self.score = 0.0
        self.steps += 1

        self.last_state = state
        self.last_action = action

        return action

    def policy(self, state, training):
        """Epsilon-greedy policy for training, greedy policy otherwise."""
        explore_prob = self.max_explore - (self.steps * self.anneal_rate)#probabilidad de exploracion decreciente
        explore = max(explore_prob, self.min_explore) > np.random.rand()

        if training and explore: #hacer exploracion
            action = np.random.randint(self.action_space_size)
        else: #hacer explotacion
            inputs = np.expand_dims(state, 0)
            qvalues = self.online_network.model(inputs) #online or evalation network predicts q-values
            #print("***##qvalues",qvalues)
            action = np.squeeze(np.argmax(qvalues, axis=-1))#need work

        return action
    
    def save(self):
        None

    # def update_target_network(self):
    #     """Update target network weights with current online network values."""
    #     variables = self.online_network.trainable_variables
    #     variables_copy = [tf.Variable(v) for v in variables]
    #     self.target_network.trainable_variables = variables_copy#need work

    # def train_network(self):
    #     """Update online network weights."""
    #     batch = self.memory.sample(self.batch_size)
    #     inputs = np.array([b["state"] for b in batch]) #####
    #     actions = np.array([b["action"] for b in batch])
    #     rewards = np.array([b["reward"] for b in batch])
    #     next_inputs = np.array([b["next_state"] for b in batch])

    #     actions_one_hot = np.eye(self.action_space_size)[actions]

    #     next_qvalues = np.squeeze(self.target_network.model(next_inputs))
    #     targets = rewards + self.discount * np.amax(next_qvalues, axis=-1)

    #     self.online_network.train_step(inputs, targets, actions_one_hot)#need work




#def main():
   
    # env = gym.make('CartPole-v1')
    # q = Qnet()
    # q_target = Qnet()
    # q_target.load_state_dict(q.state_dict())
    # memory = ReplayBuffer()

   
    # score = 0.0  
    # optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    # for n_epi in range(10000):
    #     epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
    #     s, _ = env.reset()
    #     done = False

    #     while not done:
    #         a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
    #         s_prime, r, done, truncated, info = env.step(a)
    #         done_mask = 0.0 if done else 1.0
    #         memory.put((s,a,r/100.0,s_prime, done_mask))
    #         s = s_prime

    #         score += r
    #         if done:
    #             break
            
    #     if memory.size()>2000:
    #         train(q, q_target, memory, optimizer)

    #     if n_epi%print_interval==0 and n_epi!=0:
    #         q_target.load_state_dict(q.state_dict())
    #         print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
    #                                                         n_epi, score/print_interval, memory.size(), epsilon*100))
    #         score = 0.0
    # env.close()

# if __name__ == '__main__':
#     main()
