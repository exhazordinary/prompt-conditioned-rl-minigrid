from env.minigrid_env import make_env
from models.prompt_encoder import PromptEncoder

def test_setup():
    # Initialize environment
    env = make_env("MiniGrid-GoToObject-8x8-N2-v0")
    obs, _ = env.reset()
    print("Observation keys:", obs.keys())
    print("Image shape:", obs["image"].shape)
    print("Mission:", obs["mission"])

    # Test prompt encoder
    encoder = PromptEncoder()
    prompt = obs['mission']
    embedding = encoder.encode(prompt)
    print("Prompt embedding shape:", embedding.shape)

if __name__ == "__main__":
    test_setup()