import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium
from collections import deque

class ReplayBuffer():
    ''' The replay memory for the networks, where we can sample batches of (state, action, reward, next state, done) transition tuples.
    A deque is used to favor recent memories over old ones when the size limit is reached.
    '''
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_size)

    def push(self, transition):
        ''' Add a transition in the from (s, a, r, s', d) to the buffer'''
        self.buffer.append(transition)
    
    def sample(self):
        ''' Samples random batch of transitions from the buffer '''
        batch_idxs = np.random.choice(len(self.buffer), self.batch_size, replace=True)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch_idxs])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class QNet(nn.Module):
    ''' Defines the model and its usage, such as training and running. '''

    def __init__(self, env, replay_buffer_size=15000, replay_batch_size=32, switch_every=64, epsilon=0.05, decay=0.99, discount=0.99, burn_in=64, min_epsilon=0.005):
        ''' Initializes DDQN model for a given environment and hyperparameters '''
        super().__init__()
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.replay_buffer_size = int(replay_buffer_size)
        self.replay_batch_size = int(replay_batch_size)
        self.switch_every = switch_every
        self.epsilon = epsilon
        self.decay = decay
        self.discount = discount
        self.burn_in = burn_in
        self.min_epsilon = min_epsilon

        self.Qnet = lambda : nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            # nn.GELU(),
            # nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, self.action_size)
        )

        self.target = self.Qnet()
        self.online = self.Qnet()
        self.optimizer = optim.Adam(self.online.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.replay_batch_size)

        if self.burn_in:
            print('burning in buffer...')
            self.collect_transitions()

    def collect_transitions(self):
        ''' Collect a given number of random transitions to instantiate the replay buffer '''
        transitions = 0
        while transitions < self.burn_in:
            state = self.env.reset()[0]
            done = False
            while not done and transitions < self.burn_in:
                action = self.env.action_space.sample()
                next_state, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.push((state, action, reward, next_state, done))
                state = next_state
                transitions += 1

    
    def get_action(self, state):
        '''samples action given state from online and target networks using epsilon-greedy sampling'''

        if np.random.rand() < self.epsilon:
            # stochastic choice - random action chosen
            return np.random.randint(0, self.action_size)
        else:
            # greedy choice - action maximizing online network Q-value
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)   
            with torch.no_grad():
                return self.online(state).argmax().item()
    
    def train_step(self):
        ''' sample a batch of transitions and train on it '''
        if len(self.replay_buffer) < self.replay_batch_size:
            # not enough transitions in the buffer to train - need to collect more experience
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

        # estimate Q-values using online network
        q_values = self.online(states).gather(1, actions)

        # estimate next state Q-values with Target network
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(1).unsqueeze(-1)
            next_q_values = self.target(next_states).gather(1, next_actions)
        
        # traget Q-values
        q_values_target = rewards + (self.discount * next_q_values * (1 - dones))

        # optimize over MSE loss of Q-values (online network)
        loss = nn.MSELoss()(q_values, q_values_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, n_episodes, max_steps, verbose=False):
        '''train model for n_episodes, each with at most max_steps steps. Return collected rewards for easy plotting'''
        rewards = []
        total_steps = 0

        for episode in range(n_episodes):
            state = self.env.reset()[0]
            episode_reward = 0
            for step in range(max_steps):
                # sample action and step in that direction
                action = self.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                # add transition to buffer:
                self.replay_buffer.push((state, action, reward, next_state, done))

                # take a training step
                self.train_step()

                # update states
                state = next_state
                episode_reward += reward
                total_steps += 1
                if total_steps % self.switch_every == 0:
                    self.target.load_state_dict(self.online.state_dict()) # 'switch' networks

                if done:
                    break
            
            if verbose and episode%25 == 0:
                print(f'Episode {episode:03d} --- reward = {episode_reward}')
            
            # record rewards and decay epsilon - become greedier
            rewards.append(episode_reward)
            # Save each reward to rewards.txt
            with open('rewards.txt', 'a') as f:
                f.write(f"{episode_reward}\n")

            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        
        # Save the trained model
        model_save_path = 'trained_ddqn_model.pth'
        torch.save(self.online.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')

        return rewards
        
    def run(self, n_runs=1, render=False):
        ''' Run the trained model n_runs times and return total rewards '''
        total_rewards = []

        for run in range(n_runs):
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            while not done:
                if render:
                    self.env.render()

                action = self.get_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)
            print(f'Run {run+1}/{n_runs}, Total Reward: {total_reward}')

        return total_rewards