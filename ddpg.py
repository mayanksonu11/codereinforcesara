import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
buffer_limit = 50000
tau          = 0.005 # for target network soft update

class ReplayBuffer():
    def __init__(self,max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        print("**",buffer_size)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]
    
    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):
    def __init__(self,state_space_size,action_space_size):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_space_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, action_space_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc_mu(x))
        mu = F.sigmoid(x) # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        # print(mu)
        return mu

class QNet(nn.Module):
    def __init__(self,state_space_size,action_space_size):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_space_size, 64)
        self.fc_a = nn.Linear(action_space_size,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat((h1,h2), dim=0)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
      

    

    
class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = cns.deque(maxlen=max_size)

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        print("**",buffer_size)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)

class Agent(object):
    def __init__(self,
                state_space_size,
                action_space_size,
                target_update_freq=50, #1000, #cada n steps se actualiza la target network
                discount=0.99,
                batch_size=80,
                max_explore=1,
                min_explore=0.05,
                anneal_rate=(1/5000), #100000),
                replay_memory_size=100000,
                replay_start_size= 100): #500): #10000): #despues de n steps comienza el replay
        """Set parameters, initialize network."""
        self.action_space_size = action_space_size

        self.q, self.q_target = QNet(state_space_size,action_space_size), QNet(state_space_size,action_space_size)
        self.q_target.load_state_dict(self.q.state_dict())
        self.mu, self.mu_target = MuNet(state_space_size,action_space_size), MuNet(state_space_size,action_space_size)
        self.mu_target.load_state_dict(self.mu.state_dict())

        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=lr_mu)
        self.q_optimizer  = optim.Adam(self.q.parameters(), lr=lr_q)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_space_size))

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

    def step(self, state, reward, training=True):
        """Observe state and rewards, select action.
        It is assumed that `observation` will be an object with
        a `state` vector and a `reward` float or integer. The reward
        corresponds to the action taken in the previous step.
        """
        last_state, last_action = self.last_state, self.last_action
        last_reward = reward
        state = state
        
        action = self.policy(state, training)

        if training:
            self.steps += 1
            print("## step:",self.steps)

            if last_state is not None:
                experience = {
                    "state": last_state,
                    "action": last_action,
                    "reward": last_reward,
                    "next_state": state
                }
                # print("Experience:",experience)
                self.memory.put(experience)

            if self.steps % self.replay_start_size == 0: #para acumular cierta cantidad de experiences antes de comenzar el entrenamiento
                for i in range(10):
                    self.train(self.mu, self.mu_target, self.q, self.q_target, self.memory, self.q_optimizer, self.mu_optimizer)
                    self.soft_update(self.mu, self.mu_target)
                    self.soft_update(self.q,  self.q_target)

        self.last_state = state
        self.last_action = action

        return action.detach().numpy()
        # return [1,1,1]

    def policy(self, state, training):
        """Epsilon-greedy policy for training, greedy policy otherwise."""
        state = np.array(state)
        action =  self.mu(torch.from_numpy(state).float()) 
        for i in range(len(action)):
            action[i] = max(0, action[i] + self.ou_noise()[i])

        return action

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    def train(self, mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
        batch  = memory.sample(self.batch_size)
        inputs = np.array([b["state"] for b in batch]) #####
        actions = [b["action"] for b in batch]
        rewards = np.array([b["reward"] for b in batch])
        next_inputs = np.array([b["next_state"] for b in batch])
        for i in range(len(batch)):
            s = torch.tensor(inputs[i])
            a = torch.tensor(actions[i])
            r = torch.tensor(rewards[i])
            s_prime = torch.tensor(next_inputs[i])

            target = r + gamma * q_target(s_prime, mu_target(s_prime))
            q_loss = F.smooth_l1_loss(q(s,a), target.detach())
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()
            
            mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
            mu_optimizer.zero_grad()
            mu_loss.backward()
            mu_optimizer.step()

    def save(self):
        return
    
    def put_data(self, state, action, reward, s_prime):
        None

def main():
    env = gym.make('Pendulum-v0')

    memory = ReplayBuffer()

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(10000):
        s = env.reset()
        done = False
        
        while not done:
            a = mu(torch.from_numpy(s).float()) 
            a = a.item() + ou_noise()[0]
            s_prime, r, done, info = env.step([a])
            memory.put((s,a,r/100.0,s_prime,done))
            score += r
            s = s_prime
                
        if memory.size()>2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()