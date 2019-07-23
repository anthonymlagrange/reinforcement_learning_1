from collections import deque, namedtuple
import numpy as np
import random


class ReplayBuffer:
    """A first-in-first-out buffer to store experiences."""

    
    def __init__(self, buffer_size):
        """Initializes the buffer.

        Params
        ======
            buffer_size (int): the maximum size of the buffer
        """
        self.memory = deque(maxlen=buffer_size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        
    def add(self, state, action, reward, next_state, done):
        """Adds a new experience to the buffer.
        
        Params
        ======
            state (array_like): the current state
            action (int): the action taken
            reward (float): the reward received for taking the action in the state
            next_state (array_like): the next state
            done (bool): indicates whether the episode is done or not        
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        
    def sample(self, size):
        """Randomly samples experiences from the buffer (with replacement).
        
        Params
        ======
            size (int): the size of the sample
            
        Returns
        =======
            A tuple of vectors of the components of the experiences.
        """
        experiences = random.sample(self.memory, k=size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
  
        return (states, actions, rewards, next_states, dones, None)


    def __len__(self):
        """Returns the current size of internal memory."""
        return len(self.memory)
