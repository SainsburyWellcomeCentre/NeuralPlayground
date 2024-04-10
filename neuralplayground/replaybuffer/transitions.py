class Transition(object):
    def __init__(self, prev_state, action, reward, next_state, terminate, priority=None):
        """Transition object that stores the previous state, action, reward, next state and termination flag
        Data type of each element is arbitrary except for the termination flag which is boolean
        ------
        Attributes
        prev_state:
            Previous state
        action: int
            Action taken
        reward:
            Reward received
        next_state:
            Next state
        terminate: bool
            Trial or episode termination flag,
        priority:
            Unnormalized priority to later compute sampling probabilities as in prioritize experience replay"""
        self.prev_state = prev_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminate = terminate
        self.priority = priority

    def __str__(self):
        """String representation of the transition object"""
        return_str = "Transition:\n"
        return_str += "s_prev: " + str(self.prev_state) + ", "
        return_str += "s_next: " + str(self.next_state) + ", "
        return_str += "a: " + str(self.action) + ", "
        return_str += "r: " + str(self.reward) + ", "
        return_str += "terminate: " + str(self.terminate) + ", "
        return_str += "priority: " + str(self.priority) + ", "
        return_str += "update_weights: " + str(self.update_weights)
        return return_str

    def equal(self, transition):
        """Check if two transition objects are equal by comparing their string representations"""
        if self.__str__() == transition.__str__():
            return True
        else:
            return False
