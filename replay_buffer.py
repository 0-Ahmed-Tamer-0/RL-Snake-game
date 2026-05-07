import random
from collections import deque
import torch

class ReplayBuffer:
    
    def __init__(self, capacity=10000, batch_size=32, device="cpu"):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def push(self, state, action, reward, next_state, done):
        
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors
        states      = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions     = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones       = torch.tensor(dones, dtype=torch.float32).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

    def is_ready(self):
        return len(self.memory) >= self.batch_size

