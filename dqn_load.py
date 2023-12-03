import time
import gym
from stable_baselines3 import DQN
from run_ple_utils import make_ple_env
import numpy as np
import pygame
from dqn_wrapper import PESWrapper
import os

env_id = 'ContFlappyBird-v3'
env = make_ple_env(env_id, seed=0)
env = PESWrapper(env)

models_dir = f"C:/Users/mackj/OneDrive/Desktop/DQN/models/DQN-1700919381"

model_path = f"{models_dir}/178000"

model = DQN.load(model_path, env=env)

episodes = 1
total_timesteps = 1000


# Performance Efficiency Score metric parameters
time_in_gap = 0  # Counter for time spent in the gap
total_flaps = 0  # Counter for total flaps

frame_interval = 71

bird_positions = []
frames = []
bird_safe = []
flap_positions = []  # Store the positions where the bird flaps

for ep in range(episodes):
    obs = env.reset()
    done = False
    t = 0
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        bird_y = obs[0]
        gap_top = obs[3]
        gap_bottom = obs[2]

        # Check if the bird is in the gap
        is_safe = gap_bottom < bird_y < gap_top
        bird_safe.append(is_safe)

        if is_safe:
            time_in_gap += 1
        
        if action == 0:
            total_flaps += 1  # Increment total flaps if the bird flaps this timestep
            flap_positions.append(t)

        # Capture frame at specified interval
        if t % frame_interval == 0: 
            frame = env.render(mode='rgb_array')
            frame = np.flip(frame, axis=1)  # Flip horizontally if needed
            frames.append(frame)

        # Store bird's position at every timestep
        x_position = t * frame.shape[1] // frame_interval + 70
        bird_positions.append((x_position, int(bird_y)))

        t += 1  # Increment timestep counter

        if done:
            obs = env.reset()
env.close()

# Combine frames and draw the path
combined_surface = pygame.Surface((len(frames) * frames[0].shape[1], frames[0].shape[0]))

for i, frame in enumerate(frames):
    frame_surface = pygame.surfarray.make_surface(frame).convert()
    frame_surface = pygame.transform.rotate(frame_surface, -90)  # Rotate -90 degrees
    combined_surface.blit(frame_surface, (i * frame_surface.get_width(), 0))

# Draw the bird's path on the combined surface
for i in range(1, len(bird_positions)):
    # Determine the color based on whether the bird was safe
    color = (0, 255, 0) if bird_safe[i] else (255, 0, 0)  # Green if safe, red otherwise
    pygame.draw.line(
        combined_surface, 
        color, 
        bird_positions[i - 1], 
        bird_positions[i], 
        5  # Width of the path
    )

for flap_time in flap_positions:
        x_flap_position = flap_time * frame.shape[1] // frame_interval + 70
        pygame.draw.line(
            combined_surface,
            (255, 255, 255),  # White color for flap line
            (x_flap_position, 0),
            (x_flap_position, frames[0].shape[0]),
            2  # Width of the flap line
        )

pygame.image.save(combined_surface, 'C:/Users/mackj/OneDrive/Desktop/SimPaths/dqn_path.png')
lam = 1
PES = lam * time_in_gap / t - (1-lam )* ( total_flaps / t )
print(f'Performance Efficiency Score: {PES}')
print(f'Total Timesteps: {t}')
