import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptConditionedPolicy(nn.Module):
    def __init__(self, obs_dim, prompt_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + prompt_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, obs, prompt):
        x = torch.cat([obs, prompt], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.policy_head(x), self.value_head(x)
