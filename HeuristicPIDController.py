import sys
import numpy as np
import pygame
from run_ple_utils import make_ple_env

class HeuristicPIDController:
    def __init__(self):
        # Adjust control gains
        self.Kp = 0.8  # Proportional gain
        self.Ki = 0.4  # Integral gain
        self.Kd = 0.9  # Derivative gain
        self.prev_error = 0
        self.integral_error = 0

    def control_action(self, y, y_setpoint):
        error = y_setpoint - y
        self.integral_error += error
        derivative_error = error - self.prev_error
        u = self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative_error
        self.prev_error = error
        return u

    def decide_action(self, state):
        y = state[0]  # Bird's vertical position
        y_setpoint = 0.6 * (state[2] + state[3])  # Adjust setpoint for better alignment
        u = self.control_action(y, y_setpoint)
        return 1 if u > 0 else 0

def main_HeuristicPID():
    env_id = 'ContFlappyBird-v3'
    total_timesteps = 3000
    seed = 0
    video_path = 'C:/Users/mackj/OneDrive/Desktop/SimVids/simulation_hPID.mp4'
    image_path = 'C:/Users/mackj/OneDrive/Desktop/SimPaths/simulation_hPID_path.png'
    
    frame_interval = 71
    test_env = make_ple_env(env_id, seed=seed)
    state = test_env.reset()
    controller = HeuristicPIDController()

    frames = []
    bird_positions = []
    bird_safe = []

    time_in_gap = 0
    total_flaps = 0

    for t in range(total_timesteps):
        action = controller.decide_action(state)

        bird_y = state[0]
        gap_top = state[3]
        gap_bottom = state[2]
        is_safe = gap_bottom < bird_y < gap_top
        bird_safe.append(is_safe)

        if is_safe:
            time_in_gap += 1
        
        if action == 1:
            total_flaps += 1

        state, reward, done, _ = test_env.step(action)
        if t % frame_interval == 0:
            frame = test_env.render(mode='rgb_array')
            frames.append(np.flip(frame, axis=1))
        x_position = t * frame.shape[1] // frame_interval + 70
        bird_positions.append((x_position, int(bird_y)))

        if done:
            state = test_env.reset()

    test_env.close()

    combined_surface = pygame.Surface((len(frames) * frames[0].shape[1], frames[0].shape[0]))
    for i, frame in enumerate(frames):
        frame_surface = pygame.surfarray.make_surface(frame).convert()
        frame_surface = pygame.transform.rotate(frame_surface, -90)
        combined_surface.blit(frame_surface, (i * frame_surface.get_width(), 0))

    for i in range(1, len(bird_positions)):
        color = (0, 255, 0) if bird_safe[i] else (255, 0, 0)
        pygame.draw.line(combined_surface, color, bird_positions[i - 1], bird_positions[i], 5)

    pygame.image.save(combined_surface, image_path)

    lam = 1
    PES = lam * time_in_gap / total_timesteps - (1-lam) * (total_flaps / total_timesteps)
    print(f'Performance Efficiency Score: {PES}')

if __name__ == '__main__':
    main_HeuristicPID()
