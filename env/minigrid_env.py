import gymnasium as gym
import minigrid  # Registers all MiniGrid envs

def make_env(env_id="MiniGrid-Empty-5x5-v0", render_mode="rgb_array"):
    return gym.make(env_id, render_mode=render_mode)

def list_minigrid_envs():
    env_ids = [env_id for env_id in gym.envs.registry.keys() if "MiniGrid" in env_id]
    print(f"Total MiniGrid Envs: {len(env_ids)}")
    for env_id in sorted(env_ids):
        print(env_id)

if __name__ == "__main__":
    list_minigrid_envs()