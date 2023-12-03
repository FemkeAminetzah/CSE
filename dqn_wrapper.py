from gym import Wrapper
class PESWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.time_in_gap = 0
        self.total_flaps = 0
        self.total_timesteps = 0
        self.bird_positions = []
        self.frames = []
        self.bird_safe = []
        self.frame_interval = 71

    def reset(self, **kwargs):
        self.time_in_gap = 0
        self.total_flaps = 0
        self.total_timesteps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        bird_y = state[0]
        gap_top = state[3]
        gap_bottom = state[2]
        self.bird_positions.append((self.total_timesteps, bird_y))

        is_safe = gap_bottom < bird_y < gap_top
        self.bird_safe.append(is_safe)

        if is_safe:
            self.time_in_gap += 1
        
        if action == 1:  
            self.total_flaps += 1

        self.total_timesteps += 1
        frame = self.env.render(mode='rgb_array')
        self.frames.append(frame)

        return state, reward, done, info
