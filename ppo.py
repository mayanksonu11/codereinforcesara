import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(9,256)
        self.fc_pi = nn.Linear(256,3)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        action = F.softmax(x, dim=softmax_dim)
        # action = F.sigmoid(F.relu(x))
        # print("Action:",action)
        return action
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []
        for transition in self.data:
            s, a, r, s_prime = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            # prob_a_lst.append([prob_a])
            # done_mask = 0 if done else 1
            # done_lst.append([done_mask])
            
        s,a,r,s_prime = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float)
                                        #   torch.tensor(prob_a_lst)
                                        #   torch.tensor(done_lst, dtype=torch.float), 
        self.data = []
        return s, a, r, s_prime, 
        
    def train_net(self):
        s, a, r, s_prime = self.make_batch()
        a_arg_max = []
        a_max = []
        for j in range(len(a)):
            a_arg_max.append([a[j].argmax()])
            a_max.append([a[j].max()])
        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime)
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            # print("Pi:",pi," action:",a)
            pi_a = pi.gather(1,torch.tensor(a_arg_max))
            ratio = torch.exp(torch.log(pi_a) - torch.log(torch.tensor(a_max,dtype=float)))  # a/b == exp(log(a)-log(b))
            ratio = torch.tensor(1, dtype=float)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

class Agent(object):
    def __init__(self,
                state_space_size,
                action_space_size,
                target_update_freq=50, #1000, #cada n steps se actualiza la target network
                discount=0.99,
                batch_size=32,
                max_explore=1,
                min_explore=0.05,
                anneal_rate=(1/5000), #100000),
                replay_memory_size=100000,
                replay_start_size= 50): #500): #10000): #despues de n steps comienza el replay
        """Set parameters, initialize network."""
        self.action_space_size = action_space_size
        self.steps = 0
        self.model = PPO()
        score = 0.0
        self.print_interval = 20
        self.target_update_freq = target_update_freq

    def handle_episode_start(self):
        self.last_state, self.last_action = None, None

    def step(self, state, reward, training=True):
        """Observe state and rewards, select action.
        It is assumed that `observation` will be an object with
        a `state` vector and a `reward` float or integer. The reward
        corresponds to the action taken in the previous step.
        """
        last_state, last_action = self.last_state, self.last_action
        last_reward = reward
        state = state
        state = np.array(state)
        action = self.model.pi(torch.from_numpy(state).float())

        # model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
        # s = s_prime

        if training:
            self.steps += 1
            print("## step:",self.steps)

            # if last_state is not None:
            #     experience = {
            #         "state": last_state,
            #         "action": last_action,
            #         "reward": last_reward,
            #         "next_state": state
            #     }

            #     self.memory.put(experience)

            if self.steps % self.target_update_freq == 0: #para acumular cierta cantidad de experiences antes de comenzar el entrenamiento
                self.model.train_net()

        self.last_state = state
        self.last_action = action

        return action.detach().numpy()
    
    def put_data(self, state, action, reward, s_prime):
        self.model.put_data((state, action, reward, s_prime))

    def train(self, mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
        None
    def save(self):
        return
        
def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()