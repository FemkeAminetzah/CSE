import time
import gym
from stable_baselines3 import PPO
from run_ple_utils import make_ple_env
import numpy as np
import pygame
from dqn_wrapper import PESWrapper
import os
import csv
import imageio

def main_ppo(seed):
    env_id = 'ContFlappyBird-v3'
    env = make_ple_env(env_id, seed=seed)
    env = PESWrapper(env)

    models_dir = f"C:/Users/mackj/OneDrive/Desktop/PPO/models/PPO-1705263177"  # Update the timestamp to your PPO models directory
    model_path = f"{models_dir}/120000"  # Adjust this path to the specific model you want to load

    model = PPO.load(model_path, env=env)

    episodes = 1
    total_timesteps = 3000

    # Performance Efficiency Score metric parameters
    time_in_gap = 0
    total_flaps = 0

    frame_interval = 71

    bird_positions = []
    frames = []
    bird_safe = []
    flap_positions = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        t = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            bird_y = obs[0]
            gap_top = obs[3]
            gap_bottom = obs[2]

            is_safe = gap_bottom < bird_y < gap_top
            bird_safe.append(is_safe)

            if is_safe:
                time_in_gap += 1
            
            if action == 0:
                total_flaps += 1
                flap_positions.append(t)
 
            frame = env.render(mode='rgb_array')
            frame = np.flip(frame, axis=1)
            frames.append(frame)

            x_position = t * frame.shape[1] // frame_interval + 70
            adjusted_bird_y = bird_y + 13
            bird_positions.append((x_position, int(adjusted_bird_y)))

            t += 1

            if done:
                obs = env.reset()
    env.close()

    combined_surface = pygame.Surface((len(frames) * frames[0].shape[1], frames[0].shape[0]))

    for i, frame in enumerate(frames):
        frame_surface = pygame.surfarray.make_surface(frame).convert()
        frame_surface = pygame.transform.rotate(frame_surface, -90)
        combined_surface.blit(frame_surface, (i * frame_surface.get_width(), 0))

    for i in range(1, len(bird_positions)):
        color = (255, 0, 0)
        pygame.draw.line(
            combined_surface, 
            color, 
            bird_positions[i - 1], 
            bird_positions[i], 
            5
        )


    pygame.image.save(combined_surface, 'C:/Users/mackj/OneDrive/Desktop/SimPaths/ppo_path.png') 

    writer = imageio.get_writer('C:/Users/mackj/OneDrive/Desktop/SimVids/simulation_event_dependent.mp4', fps=60)
    for frame in frames:
        flipped_frame = np.flip(frame, axis=1)
        writer.append_data(flipped_frame)
    writer.close()

    lam = 1
    PES = lam * time_in_gap/total_timesteps
    print(PES)
    return PES

if __name__ == '__main__':
    main_ppo(seed=7)
    """ with open('ppo_controller_pm.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'PM'])

        for i in range(150):
            pes = main_ppo(seed=i)
            writer.writerow([i, pes]) """
