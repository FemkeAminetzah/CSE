import argparse
import os, sys
import numpy as np
import random
import time
sys.path.append("FlappyBird_agents_upgraded")
sys.path.append("FlappyBird_environment_upgraded")
from run_ple_utils import make_ple_env

class HeuristicPIDController:
    def __init__(self):
        # Control gains
        self.Kp = 1.0
        self.Ki = 0.5
        self.Kd = 1.0
        # Previous error and integral of error
        self.prev_error = 0
        self.integral_error = 0

        self.flaps = 0
        self.pipe_collisions = 0

    def control_action(self, y, y_setpoint):
        """Compute the control action based on the error, its integral, and its derivative."""
        error = y_setpoint - y
        self.integral_error += error
        d_error = error - self.prev_error
        # Control law
        u = self.Kp * error + self.Ki * self.integral_error + self.Kd * d_error
        self.prev_error = error
        return u

    def decide_action(self, state):
        """Decide whether to flap based on the control action."""
        y = state[0]  # Bird's vertical position
        y_setpoint = 0.5 * (state[2] + state[3])  # Midpoint between pipes
        u = self.control_action(y, y_setpoint)
        if u > 0:
            self.flaps += 1
        return 1 if u > 0 else 0  # Flap if control action is positive

def main_HeuristicPID():
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

    for s in range(100, 120):
        test_env = make_ple_env(args.test_env, seed=s)
        state = test_env.reset()
        total_return = 0

        t = 0
        controller = HeuristicPIDController()
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
    main_HeuristicPID()
