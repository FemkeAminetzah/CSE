import argparse
import logging
import os, sys
import csv
import numpy as np
import random
import time
import imageio
import pygame
sys.path.append("FlappyBird_agents_upgraded")
sys.path.append("FlappyBird_environment_upgraded")
from run_ple_utils import make_ple_env

def compute_discrete_action(state, game_instance):
    bird_y = state[0]
    gap_bottom = state[2]
    gap_top = state[3]
    # Calculate the halfway point between the top and bottom of the pipe gap
    halfway_point = (gap_top + gap_bottom) / 2
    # Define a range around the halfway point where the bird should maintain altitude
    tolerance = (gap_bottom - gap_top) * 0.2  # Adjust the 0.2 value as needed
    lower_bound = halfway_point + tolerance
    upper_bound = halfway_point - tolerance
    
    # Ascend if the bird is above the upper bound
    if bird_y > upper_bound:
        return 0  # Action for ascent

    # Descend if the bird is below the lower bound
    elif bird_y < lower_bound:
        return 1  # Action for descent
    
    # Maintain altitude if the bird is within the bounds
    else:
        return 2  # Action for maintaining altitude
    
def main_reactive(seed):
    # Parameters
    env_id = 'ContFlappyBird-v3'
    total_timesteps = 600
    video_path = 'C:/Users/mackj/OneDrive/Desktop/SimVids/simulation_event_dependent.mp4'
    image_path = 'C:/Users/mackj/OneDrive/Desktop/SimPaths/reactive_path.png'
    
    frame_interval = 71  # Capture every 71st frame

    # Performance Efficiency Score metric parameters
    time_in_gap = 0  # Counter for time spent in the gap
    total_flaps = 0  # Counter for total flaps

    test_env = make_ple_env(env_id, seed=seed)
    state = test_env.reset()

    frames = []  # Store frames for video
    bird_positions = []  # Store the bird's positions to draw the path
    bird_safe = []
    flap_positions = []  # Store the positions where the bird flaps
    
    action_change_points = []  # Store points of action change
    last_action = None


    for t in range(total_timesteps):
        
        action = compute_discrete_action(state, test_env)
        
        if action != last_action:
            action_change_points.append((t, action))
            last_action = action

        bird_y = state[0]
        gap_top = state[3]
        gap_bottom = state[2]

        is_safe = gap_bottom < bird_y < gap_top
        bird_safe.append(is_safe)

        if is_safe:
            time_in_gap += 1

        if action == 0:
            total_flaps += 1
            flap_positions.append(t)
        
        state, reward, done, _ = test_env.step(action)

        if t % frame_interval == 0:
            frame = test_env.render(mode='rgb_array')
            frame = np.flip(frame, axis=1)  # Flip horizontally
            frames.append(frame)
            time.sleep(0.1)

        # Store bird's position at every timestep
        #bird_positions.append((t * frame.shape[1] // frame_interval, int(bird_y)))  # Scale x-coordinate based on frame interval

        x_position = t * frame.shape[1] // frame_interval + 70
        adjusted_bird_y = bird_y + 13
        bird_positions.append((x_position, int(adjusted_bird_y)))

        if done:
            state = test_env.reset()


    test_env.close()

    # Combine frames and draw the path
    combined_surface = pygame.Surface((len(frames) * frames[0].shape[1], frames[0].shape[0]))

    for i, frame in enumerate(frames):
        frame_surface = pygame.surfarray.make_surface(frame).convert()
        frame_surface = pygame.transform.rotate(frame_surface, -90)  # Rotate -90 degrees
        combined_surface.blit(frame_surface, (i * frame_surface.get_width(), 0))

    # Draw the bird's path on the combined surface
    for i in range(1, len(bird_positions)):
        # Determine the color based on whether the bird was safe
        color = (255, 0, 0) #if bird_safe[i] else (255, 0, 0)  # Green if safe, red otherwise
        pygame.draw.line(
            combined_surface, 
            color, 
            bird_positions[i - 1], 
            bird_positions[i], 
            5  # Width of the path
        ) 
    # Draw vertical lines for action changes
    """ for point, action in action_change_points:
        x_position = point * frames[0].shape[1] // frame_interval + 70
        if action == 0:  # Ascend
            color = (128, 0, 128)  # Purple
        elif action == 1:  # Descend
            color = (0, 0, 255)  # Blue
        else:  # Maintain
            color = (255, 255, 255)  # White

        pygame.draw.line(
            combined_surface,
            color,
            (x_position, 0),
            (x_position, frames[0].shape[0]),
            2  # Width of the vertical line
        )
 """


    pygame.image.save(combined_surface, image_path)


    writer = imageio.get_writer(video_path, fps=60)
    for frame in frames:
        flipped_frame = np.flip(frame, axis=1)
        writer.append_data(flipped_frame)
    writer.close()
    lam = 1
    PES = lam * time_in_gap/total_timesteps
    print(PES)
    return PES



if __name__ == '__main__':
    main_reactive(seed=2)
    """ with open('reactive_controller_pm.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'PM'])

        for i in range(150):
            pes = main_reactive(seed=i)
            writer.writerow([i, pes]) """