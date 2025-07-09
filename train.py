import torch
import numpy as np
from torch.distributions import Categorical

from env.minigrid_env import make_env
from models.prompt_encoder import PromptEncoder
from models.policy import PromptConditionedPolicy
from utils.preprocess import preprocess_obs

# --- Config ---
env_id = "MiniGrid-GoToObject-8x8-N2-v0"
render_mode = None
steps_per_epoch = 2048
train_iters = 4
gamma = 0.99
clip_eps = 0.2
lr = 3e-4
epochs = 500

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = make_env(env_id, render_mode=render_mode)
obs, _ = env.reset()
obs_dim = np.prod(obs["image"].shape)
action_dim = env.action_space.n
prompt_dim = 768
MAX_EPISODE_STEPS = getattr(env, "_max_episode_steps", 100)

encoder = PromptEncoder()
policy = PromptConditionedPolicy(obs_dim, prompt_dim, action_dim).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

def discount_rewards(rewards, gamma):
    discounted = []
    r = 0
    for reward in reversed(rewards):
        r = reward + gamma * r
        discounted.insert(0, r)
    return discounted

# --- Training Loop ---
for epoch in range(epochs):
    obs_buf, act_buf, logp_buf, val_buf, rew_buf = [], [], [], [], []
    ep_rewards = []

    obs, _ = env.reset()
    prompt = obs["mission"]
    prompt_vec = encoder.encode(prompt).to(device).detach()
    print(f"[Epoch {epoch}] Mission Prompt: {prompt}")

    for step in range(steps_per_epoch):
        obs_tensor = preprocess_obs(obs).to(device)
        logits, value = policy(obs_tensor, prompt_vec)
        probs = Categorical(logits=logits)
        action = probs.sample()

        next_obs, reward, done, _, _ = env.step(action.item())

        obs_buf.append(obs_tensor.squeeze(0))
        act_buf.append(action)
        logp_buf.append(probs.log_prob(action))
        val_buf.append(value.squeeze(0))
        rew_buf.append(reward)

        obs = next_obs
        ep_rewards.append(reward)

        if done:
            obs, _ = env.reset()
            prompt = obs["mission"]
            prompt_vec = encoder.encode(prompt).to(device).detach()
            print(f"[Step {step}] New Mission Prompt: {prompt}")

    # --- Compute returns and advantages ---
    returns = torch.tensor(discount_rewards(rew_buf, gamma)).to(device)
    values = torch.stack(val_buf).to(device).squeeze()
    advs = returns - values.detach()

    obs_batch = torch.stack(obs_buf).to(device).detach()
    act_batch = torch.stack(act_buf).to(device).detach()
    logp_old_batch = torch.stack(logp_buf).to(device).detach()
    returns = returns.detach()
    advs = advs.detach()

    # --- PPO Update ---
    for _ in range(train_iters):
        logits, value = policy(obs_batch, prompt_vec.expand(len(obs_batch), -1))
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_batch)
        ratio = torch.exp(logp - logp_old_batch)

        clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs
        loss_pi = -(torch.min(ratio * advs, clip_adv)).mean()
        loss_v = ((returns - value.squeeze()) ** 2).mean()
        loss = loss_pi + 0.5 * loss_v

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_reward = np.sum(ep_rewards) / (steps_per_epoch / MAX_EPISODE_STEPS)
    print(f"✅ Epoch {epoch}, Avg Reward: {avg_reward:.3f}")

# --- Save the trained model ---
save_path = "policy.pth"
torch.save(policy.state_dict(), save_path)
print(f"✅ Trained policy saved to: {save_path}")
