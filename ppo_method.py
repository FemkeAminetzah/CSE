import os
import time
import gym
import sys
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

sys.path.append("path/to/FlappyBird_agents_upgraded")
sys.path.append("path/to/FlappyBird_environment_upgraded")
from run_ple_utils import make_ple_env
import numpy as np
import pygame
from dqn_wrapper import PESWrapper

# Initialize Weights & Biases
wandb.init(project="flappybird_ppo", entity="jupiter-thesis")

# Path to save models and logs
log_dir = f"C:/Users/mackj/OneDrive/Desktop/PPO/logs/PPO-{int(time.time())}"
models_dir = f"C:/Users/mackj/OneDrive/Desktop/PPO/models/PPO-{int(time.time())}"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Create the environment
env_id = 'ContFlappyBird-v3'
env = make_ple_env(env_id, seed=0)
env = PESWrapper(env)

# Create the PPO model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

# Callbacks
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir,
                                         name_prefix='ppo_model')
eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=500,
                             deterministic=True, render=False)

# Number of timesteps for training
TIMESTEPS = 2000

# Train the model
for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    # Save the model
    model.save(f"{models_dir}/{TIMESTEPS*i}")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

# Log metrics to Weights & Biases
wandb.log({'mean_reward': mean_reward, 'std_reward': std_reward})
wandb.finish()
