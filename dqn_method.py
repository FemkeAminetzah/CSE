import os
import time
import gym
from gym import Wrapper
import sys
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
sys.path.append("path/to/ContinuousSubmarine_agents")
sys.path.append("path/to/ContinuousSubmarineEnvironment")
from run_ple_utils import make_ple_env
import numpy as np
import pygame
from dqn_wrapper import PESWrapper

log_dir = f"C:/Users/mackj/OneDrive/Desktop/DQN/logs/DQN-{int(time.time())}"
models_dir = f"C:/Users/mackj/OneDrive/Desktop/DQN/models/DQN-{int(time.time())}"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# Create the environment
env_id = 'ContFlappyBird-v3'
env = make_ple_env(env_id, seed=0)
env = PESWrapper(env)

# Create the DQN model
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

# Callbacks
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir,
                                         name_prefix='dqn_model')
eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=500,
                             deterministic=True, render=False)
TIMESTEPS = 2000
# Train the model
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    # Save the final model
    model.save(f"{models_dir}/{TIMESTEPS*i}")



# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

