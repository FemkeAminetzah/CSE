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
sys.path.append("path/to/FlappyBird_agents_upgraded")
sys.path.append("path/to/FlappyBird_environment_upgraded")
from run_ple_utils import make_ple_env
import numpy as np
import pygame
from dqn_wrapper import PESWrapper

# Initialize Weights & Biases
wandb.init(project="flappybird_dqn", entity="jupiter-thesis")

# Path to save models and logs
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
TIMESTEPS = 100
# Train the model
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    # Save the final model
    model.save(f"{models_dir}/{TIMESTEPS*i}")



# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std Reward: {std_reward}")

# Reset the environment
obs = env.reset()
# Rendering and PES calculation
#try:
#    print("Rendering and calculating PES...")
#    for t in range(1000):
#        action, _states = model.predict(obs, deterministic=True)
#        obs, rewards, dones, info = env.step(action)
#        if t % env.frame_interval == 0:
#            frame = env.render(mode='rgb_array')
#           frame = np.flip(frame, axis=1)  # Flip horizontally
#            env.frames.append(frame)
#
#        x_position = t * frame.shape[1] // env.frame_interval + 70
#        env.bird_positions.append((x_position, int(obs[0])))
#        if dones:
#            obs = env.reset()
#
#    env.close()
#
    # Combine frames and draw the path
#    combined_surface = pygame.Surface((len(env.frames) * env.frames[0].shape[1], env.frames[0].shape[0]))
#
#    for i, frame in enumerate(env.frames):
#        frame_surface = pygame.surfarray.make_surface(frame).convert()
#        frame_surface = pygame.transform.rotate(frame_surface, -90)  # Rotate -90 degrees
#        combined_surface.blit(frame_surface, (i * frame_surface.get_width(), 0))

    # Draw the bird's path on the combined surface
#    for i in range(1, len(env.bird_positions)):
        # Determine the color based on whether the bird was safe
#        color = (0, 255, 0) if env.bird_safe[i] else (255, 0, 0)  # Green if safe, red otherwise
#        pygame.draw.line(
#            combined_surface, 
#            color, 
#            env.bird_positions[i - 1], 
#            env.bird_positions[i], 
#            5  # Width of the path
#        )
#    pygame.image.save(combined_surface, 'C:/Users/mackj/OneDrive/Desktop/SimPaths/dqn_path.png')
#    pes = env.get_pes()
#    print(f"Performance Efficiency Score: {pes}")
#
#except Exception as e:
#    print(f"An error occurred: {e}")

# Log metrics to Weights & Biases
wandb.log({'mean_reward': mean_reward, 'std_reward': std_reward})
wandb.finish()
