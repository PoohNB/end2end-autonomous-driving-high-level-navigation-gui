def filter_8bit(steer,throttle):
    # Normalize steer to 0-255 and back to -1 to 1 with 8-bit precision
    new_steer = ((round(((steer+1)/2) * 255)/255)*2)-1
    # Normalize throttle to 0-255 and back to 0 to 1 with 8-bit precision
    new_throttle = (round(throttle*255)/255)

    return new_steer,new_throttle


class ActionBase:
    def __init__(self,activate_filter_8bit=False):
        self.activate_filter_8bit = activate_filter_8bit

    def _process(self,action):
        return action

    def __call__(self, action):

        action = self._process(action)
        if self.activate_filter_8bit:
            action = filter_8bit(*action)

        return float(action[0]), float(action[1]), False


# change max steer , throttle
class LimitAction(ActionBase):
    def __init__(self, 
                 throttle_range :tuple,
                 max_steer = 0.8,
                 activate_filter_8bit=False):
        super().__init__(activate_filter_8bit)
        assert 0 <= throttle_range[0] < throttle_range[1] <= 1, "Throttle range must be within (0,1) and start < end"
        assert max_steer <=1, "max_steer have to <= 1"
        self.throttle_range =throttle_range
        self.max_steer = max_steer

    def _process(self, action):

        steer, throttle = action

        new_steer = steer * self.max_steer
        new_throttle = throttle * (self.throttle_range[1] - self.throttle_range[0]) + self.throttle_range[0]

        return new_steer, new_throttle

    
class DiscretizeAction(ActionBase):

    def __init__(self, steer_list,throttle_list,steer_still_range,activate_filter_8bit=False):
        super().__init__(activate_filter_8bit)
        self.steer_list = steer_list
        self.throttle_list = throttle_list
        self.steer_still_range = steer_still_range
        self.previous_steer=0

    def _process(self,action):
        # this convert continuous action to discret
        # self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)

        steer_list = self.steer_list
        throttle_list = self.throttle_list
        steer,throttle = action

        steer_range = (2/len(steer_list))-1
        throttle_range = 1/len(throttle_list)

        for i in range(len(steer_list)):
            if steer < (i+1)*steer_range:
                new_steer = steer_list[i]

        for i in range(len(throttle_list)):
            if throttle < (i+1)*throttle_range:
                new_throttle = throttle_list[i]
        
        if abs(new_steer - self.previous_steer) < self.steer_still_range:
            new_steer = self.previous_steer
        
        self.previous_steer = new_steer
        return float(new_steer),float(new_throttle),0

