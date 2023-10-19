import argparse
import os, sys
import numpy as np
import random
import time
sys.path.append("FlappyBird_agents_upgraded")
sys.path.append("FlappyBird_environment_upgraded")
from run_ple_utils import make_ple_env
from gym import envs
#all_envs = envs.registry.all()
#env_ids = [env_spec.id for env_spec in all_envs]
#print(env_ids)
class iPIDController:
    def __init__(self, alpha, h):
        # Control gains
        self.Kp = 1.0
        self.Kd = 1.0
        # Algebraic estimation parameters
        self.alpha = alpha
        self.h = h
        # Previous error and estimation
        self.prev_error = 0
        self.prev_estimation = 0
        # Error monitoring
        self.cumulative_error = 0
        self.error_threshold = 0.5  # Threshold to adjust gains

        self.flaps = 0
        self.pipe_collisions = 0

    def algebraic_estimation(self, y):
        """Estimate the derivative of the output using algebraic estimation."""
        estimation = (y - self.prev_estimation) / self.h
        self.prev_estimation = y
        return estimation

    def adjust_gains(self):
        """Adaptively adjust control gains based on cumulative error."""
        if abs(self.cumulative_error) > self.error_threshold:
            self.Kp += 0.1
            self.Kd += 0.1
        elif abs(self.cumulative_error) < -self.error_threshold:
            self.Kp -= 0.1
            self.Kd -= 0.1

    def control_action(self, y, y_setpoint):
        """Compute the control action based on the error and its derivative."""
        error = y_setpoint - y
        self.cumulative_error += error
        d_error = (error - self.prev_error) / self.h
        # Algebraic estimation of the derivative
        d_y = self.algebraic_estimation(y)
        # Control law
        u = self.Kp * error + self.Kd * (d_error - self.alpha * d_y)
        self.prev_error = error
        # Adjust gains adaptively
        self.adjust_gains()
        return u

    def decide_action(self, state):
        """Decide whether to flap based on the control action."""
        y = state[0]  # Bird's vertical position
        y_setpoint = 0.9 * (state[2] + state[3])  # Midpoint between pipes
        u = self.control_action(y, y_setpoint)
        if u > 0:
            self.flaps += 1
        return 1 if u > 0 else 0  # Flap if control action is positive


def main_iPID():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_env', help='test environment ID', default='ContFlappyBird-v3')
    parser.add_argument('--total_timesteps', help='Total number of env steps', type=int, default=int(2e5))
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--logdir', default='C:/Users/mackj/OneDrive/Desktop/ED_CONTROL',
                        help='directory where logs are stored')
    parser.add_argument('--show_interval', type=int, default=1,
                        help='Env is rendered every n-th episode. 0 = no rendering')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Define the iPID parameters
    Kp = 1.0  # Proportional gain
    Kd = 0.5  # Derivative gain

    for s in range(100, 120):
        test_env = make_ple_env(args.test_env, seed=s)
        state = test_env.reset()
        total_return = 0

        t = 0
        controller = iPIDController(alpha=0.5, h=0.1)
        while t < args.total_timesteps:
            t += 1
            if args.show_interval > 0:
                test_env.render()
                time.sleep(0.01)
            action = controller.decide_action(state)
            state, reward, dones, _ = test_env.step(action)
            total_return += reward
            

        test_env.close()


if __name__ == '__main__':
    main_iPID()
