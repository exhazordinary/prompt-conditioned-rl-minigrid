import time
import torch
from torch.distributions import Categorical

from env.minigrid_env import make_env
from models.prompt_encoder import PromptEncoder
from models.policy import PromptConditionedPolicy
from utils.preprocess import preprocess_obs

# --- Config ---
env_id = "MiniGrid-GoToObject-8x8-N2-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load environment ---
env = make_env(env_id, render_mode="human")
obs, _ = env.reset()
prompt = obs["mission"]
print("Mission:", prompt)

# --- Recreate models ---
obs_dim = obs["image"].size
action_dim = env.action_space.n
prompt_dim = 768

encoder = PromptEncoder()
policy = PromptConditionedPolicy(obs_dim, prompt_dim, action_dim).to(device)

# --- Load trained weights ---
policy.load_state_dict(torch.load("policy.pth"))
policy.eval()

# --- Encode prompt vector for current mission ---
prompt_vec = encoder.encode(prompt).to(device).detach()

# --- Run one episode ---
done = False
total_reward = 0

while not done:
    obs_tensor = preprocess_obs(obs).to(device)
    with torch.no_grad():
        logits, _ = policy(obs_tensor, prompt_vec)
        action = torch.argmax(logits, dim=-1).item()
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()
    time.sleep(0.2)

print(f"✅ Total reward: {total_reward}")
