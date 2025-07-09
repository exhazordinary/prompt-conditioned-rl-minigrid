# MiniGrid RL Project

This project provides a reinforcement learning (RL) framework for training and evaluating agents in MiniGrid environments using Gymnasium. It includes scripts for running environments, training agents, evaluating performance, and supporting model and utility code.

## Features

- Supports MiniGrid environments via Gymnasium
- Modular code for training and evaluation
- Customizable models and preprocessing utilities

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd prompt_rl_project
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install MiniGrid environments:**
   ```bash
   pip install gymnasium-minigrid
   ```

   If you encounter issues, ensure you are using the correct Python environment.

## Usage

### List available MiniGrid environments

To list all available MiniGrid environments:
```bash
python main.py
```

### Run training

To train an agent:
```bash
python train.py
```

### Run evaluation

To evaluate a trained agent:
```bash
python evaluate.py
```

## Project Structure

```
prompt_rl_project/
│
├── main.py                  # Entry point, environment listing or demo
├── train.py                 # Training script for RL agents
├── evaluate.py              # Evaluation script for trained agents
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
├── models/                  # Model definitions
│   ├── policy.py
│   └── prompt_encoder.py
│
├── utils/                   # Utility functions
│   └── preprocess.py
│
├── env/                     # Custom environment wrappers
│   └── minigrid_env.py
│
├── prompts/                 # Prompt or scenario files (if any)
│
└── .gitignore               # Git ignore file
```

## Requirements

- Python 3.8+
- gymnasium
- gymnasium-minigrid
- Other dependencies listed in `requirements.txt`

## Notes

- Ensure you have gymnasium-minigrid installed to use MiniGrid environments.
- Modify `main.py`, `train.py`, and `evaluate.py` as needed for your experiments.
