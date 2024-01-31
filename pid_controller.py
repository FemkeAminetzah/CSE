import sys
import numpy as np
import pygame
from run_ple_utils import make_ple_env
import csv

class PIDController:
    """
    Proportional-Integral-Derivative (PID) Controller.
    """

    def __init__(self, kp, ki, kd, set_point):
        """
        Initialize the PID Controller.

        Parameters:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        set_point (float): The desired value that the controller will try to maintain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.set_point = set_point
        self.integral = 0.0
        self.previous_error = 0.0
        self.dt = 1/30
    
    def update(self, current_value):
        """
        Update the controller with the current value.
        """
        error = self.set_point - current_value

        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error * self.dt
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.previous_error) / self.dt
        self.previous_error = error

        # Control action
        control_action = proportional + integral + derivative

        # Map control action to discrete game actions
        if control_action > 1:
            return 1  # Action for descent
        elif control_action < -0.5:
            return 0  # Action for ascent
        else:
            return 2 # Action for maintaining altitude

def main_pid_control(seed):
    env_id = 'ContFlappyBird-v3'
    total_timesteps = 230
    video_path = 'C:/Users/mackj/OneDrive/Desktop/SimVids/simulation_PID.mp4'
    image_path = 'C:/Users/mackj/OneDrive/Desktop/SimPaths/simulation_PID_path.png'
    
    frame_interval = 71
    test_env = make_ple_env(env_id, seed=seed)
    state = test_env.reset()
    

    frames = []
    bird_positions = []
    bird_safe = []
    time_in_gap = 0

    pid_controller = PIDController(kp=0.5, ki=0.05, kd=0.07, set_point=0)

    for t in range(total_timesteps):
        bird_y = state[0]
        gap_top = state[3]
        gap_bottom = state[2]
        set_point = (gap_bottom + gap_top) / 2
        pid_controller.set_point = set_point

        action = pid_controller.update(bird_y)

        
        is_safe = gap_bottom < bird_y < gap_top
        bird_safe.append(is_safe)

        if is_safe:
            time_in_gap += 1

        state, reward, done, _ = test_env.step(action)
        if t % frame_interval == 0:
            frame = test_env.render(mode='rgb_array')
            frames.append(np.flip(frame, axis=1))
        x_position = t * frame.shape[1] // frame_interval + 70
        bird_positions.append((x_position, int(bird_y)))

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
        color = (255, 0, 0) if bird_safe[i] else (255, 0, 0)  # Red for both safe and unsafe for simplicity
        pygame.draw.line(
            combined_surface, 
            color, 
            bird_positions[i - 1], 
            bird_positions[i], 
            5  # Width of the path
        )

    pygame.image.save(combined_surface, image_path) 

    lam = 1
    PES = lam * time_in_gap / total_timesteps 
    print(PES)
    return PES

if __name__ == '__main__':
    main_pid_control(seed=1)
    """ with open('pid_controller_pm.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'PM'])

        for i in range(150):
            pes = main_pid_control(seed=i)
            writer.writerow([i, pes])
 """