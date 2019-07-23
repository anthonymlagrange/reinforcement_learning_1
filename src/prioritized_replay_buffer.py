from collections import namedtuple
import math
import numpy as np
import random


##################################################
##################################################
##################################################


class PrioritizedReplayBuffer:
    """A first-in-first-out prioritized buffer to store experiences."""

    
    def __init__(self, buffer_size, alpha=1, minimum_weight=1e-3):
        """Initializes the buffer.

        Params
        ======
            buffer_size (int): the maximum size of buffer
            alpha (float): the priority dampening effect
            minimum_weight (float): the minimum weight that can be given to an experience
        """
        self.memory_node_controller = NodeController(capacity=buffer_size)
        self.memory_root = None
        
        self.alpha = alpha
        self.minimum_weight = minimum_weight
        
        self.experience_tuple = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    
    def calculate_weight_from_raw_weight(self, raw_weight):
        """Calculates an experience's weight from a raw weight.
        
        Params
        ======
        raw_weight (float): the raw weight
        
        Returns
        =======
        The experience's weight.
        """
        return (abs(raw_weight) + self.minimum_weight) ** self.alpha


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
        experience = self.experience_tuple(state, action, reward, next_state, done)

        if not self.memory_root:
            raw_weight = 0
            
            node_to_add = Node(self.memory_node_controller, weight = self.calculate_weight_from_raw_weight(raw_weight=0), payload = experience)
            self.memory_root = node_to_add
            
        else:
            node_to_add = Node(self.memory_node_controller, weight = self._get_next_initial_weight(), payload = experience)
            self.memory_root = self.memory_root.add(node_to_add)


    def sample(self, size):
        """Randomly samples experiences from the buffer (with replacement).
        
        Params
        ======
            size (int): the size of the sample
            
        Returns
        =======
            A tuple of vectors of the components of the experiences together 
            with a list of the experiences' nodes (see below).
        """
        sampled_nodes = self.memory_root.sample(size=size)
        
        experiences = [n.payload for n in sampled_nodes]
        
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones, sampled_nodes)


    def __len__(self):
        """Returns the current size of internal memory."""
        return min(self.memory_node_controller.next_index, self.memory_node_controller.capacity)


    def _get_next_initial_weight(self):
        """A helper method that determines the initial weight when a new 
        experience is added to the buffer."""
        return self.memory_root.get_max_of_weights_overall()


##################################################
##################################################
##################################################


class NodeController:
    """An object of this class keeps state of a collection of nodes that make up a 
    specially constructed binary tree.

    The tree has a fixed capacity. Once that capacity has been reached, adding 
    a new node displaces the oldest node instead of making the tree bigger.

    For additional details, see the Node class below.
 """    
    
    def __init__(self, capacity):
        """Initializes a node controller object for a specially constructed binary tree.
        
        Params
        ======
        capacity (int): The capacity of the binary tree.
        """
        
        self.capacity = capacity        
        self.next_index = 0
        
    def __repr__(self):
        """Returns a user-friendly representation of the node controller."""
        
        out = f'hex(id(self))::: capacity: {self.capacity}; next_index: {self.next_index}'
        return out


##################################################
##################################################
##################################################


class Node:
    """Each node keeps track of the following attributes:
        * own index, minimum child index
            * This makes it possible to efficiently find the oldest node in the tree
              so that if the tree has reached capacity, that node can be replaced next.
        * own weight, sum of child weights
            * This makes it possible to efficiently perform weighted random sampling
              using the law of total probability.
        * maximum of child weights
            * This makes it possible to always know the maximum currently assigned weight.
              This is needed when assigning an initial weight to a new node.
        * minimum depth down
            * This makes it possible to efficiently navigate to the next best location to 
              insert a node before the tree has reached capacity.
        * the payload
    """
        
    ##################################################

    
    DEFAULT_MIN_CHILD_INDEX = math.inf
    VERBOSE = False
    
    
    ##################################################
    
    
    def __init__(self, node_controller, weight, payload=None):   
        """Initializes a node.
        
        Params
        ======
            weight (float): the (positive) weight to associate with the node
            payload (any): the node's payload
        """
        
        self.node_controller = node_controller
        
        assert weight > 0
        
        self.parent = None 
        
        self.left_child = None
        self.right_child = None

        self.own_index = self.node_controller.next_index  # For forgetting.
        self.node_controller.next_index += 1        
        self.min_child_index = Node.DEFAULT_MIN_CHILD_INDEX

        self.own_weight = weight  # For sampling.
        self.sum_of_child_weights = 0
        self.max_of_child_weights = 0
        
        self.min_depth_down = 0  # For filling.        
        
        self.payload = payload        
        
        self.previously_been_added = False
        self.has_been_replaced = False        

        
    def get_root(self):
        """Returns the root node."""
        temp = self
        while temp.parent:
            temp = temp.parent            
        return temp
        
        
    def is_terminal(self):
        """Returns True if the node is terminal (has no children) and False otherwise."""
        return not self.left_child and not self.right_child
    
    
    def is_unbalanced(self):
        """Returns True if the node is unbalanced (has a single child, the left child) and 
        False otherwise"""
        return self.left_child and not self.right_child

    
    def is_filled(self):
        """Returns True if the node is filled (has both a left and a right child) and False
        otherwise."""
        return self.left_child and self.right_child

    
    def get_max_of_weights_overall(self):
        """Returns the maximum of the node and all its descendents' weights."""
        return max(self.own_weight, self.max_of_child_weights)

    
    def add(self, child_node):
        """Adds a child node to the node.
        
        Params
        ======
            child_node (Node): The child node to add to the node.
            
        Returns
        =======
            The root node.
        """
        assert isinstance(child_node, Node)
        assert not child_node.previously_been_added
        assert self.node_controller is child_node.node_controller

        if self.node_controller.next_index <= self.node_controller.capacity:
            # Here navigate to the smallest depth and add as new there.            
            self._add_while_in_capacity(child_node)            
        else:
            # Here navigate to the smallest index and replace there.
            self._add_while_above_capacity(child_node)
        
        child_node.previously_been_added = True
        
        # Return the (possibly new) root:
        return child_node.get_root()
                
        
    def update_weight(self, weight):
        """Update the node's weight.
        
        Params
        ======
        weight (float): the node's new (positive) weight
        """
        
        assert weight > 0
        
        self.own_weight = weight
        
        temp = self
        
        while temp.parent:
            temp = temp.parent                                
            
            left_sum_of_child_weights_contribution = temp.left_child._get_sum_of_weights_overall() if temp.left_child else 0
            right_sum_of_child_weights_contribution = temp.right_child._get_sum_of_weights_overall() if temp.right_child else 0
            temp.sum_of_child_weights = left_sum_of_child_weights_contribution + right_sum_of_child_weights_contribution
            
            left_max_of_child_weights_contribution = temp.left_child.get_max_of_weights_overall() if temp.left_child else 0
            right_max_of_child_weights_contribution = temp.right_child.get_max_of_weights_overall() if temp.right_child else 0
            temp.max_of_child_weights = max(left_max_of_child_weights_contribution, right_max_of_child_weights_contribution)
                
        
    def sample(self, size=1):    
        """Performs weight-based random sampling with replacement of the node and 
        its descendents.
        
        Params
        ======
        size (int): the size of the sample
        
        Returns
        =======
        The random sample of nodes.
        """
        assert size >= 1        
        batch = [self._sample_single() for i in range(size)]                        
        return batch
    

    ##################################################
        
        
    def _get_min_index_overall(self):
        """A helper method that returns the minimum of the node and its descendents' indexes."""
        return min(self.own_index, self.min_child_index)

    
    def _get_sum_of_weights_overall(self):
        """A helper method that returns the sum of the node and its descendents' weights."""
        return self.own_weight + self.sum_of_child_weights
        
    
    def _add_while_in_capacity(self, child_node):
        """A helper method that adds a child node while the tree is within capacity."""

        if self.min_depth_down > 0:
            # Find an appropriate terminal node.
            
            if self.is_unbalanced() or self.left_child.min_depth_down > self.right_child.min_depth_down:  # Secondarily biased to the left.
                self.right_child._add_while_in_capacity(child_node)
            else:
                self.left_child._add_while_in_capacity(child_node)
        else:
            # Found an appropriate terminal node.
            assert not self.is_filled()

            if not self.left_child:
                self.left_child = child_node
            else:
                self.right_child = child_node
                
            child_node.parent = self
            
            self._refresh_upwards()


    def _replace_with(self, replacement_node):
        """A helper method that replaces a node. This is typically used once 
        the tree is at capacity."""

        replacement_node.parent = self.parent
        
        if replacement_node.parent:                
            if self is self.parent.left_child:
                replacement_node.parent.left_child = replacement_node
            elif self is self.parent.right_child:
                replacement_node.parent.right_child = replacement_node

        replacement_node.left_child = self.left_child
        if replacement_node.left_child:
            replacement_node.left_child.parent = replacement_node
        replacement_node.right_child = self.right_child
        if replacement_node.right_child:
            replacement_node.right_child.parent = replacement_node

        replacement_node.min_child_index = self.min_child_index

        replacement_node.sum_of_child_weights = self.sum_of_child_weights

        replacement_node.min_depth_down = self.min_depth_down  # For filling.        
       
        self.has_been_replaced = True


    def _add_while_above_capacity(self, child_node):
        """A helper method that adds a child node while the tree is at capacity."""

        if self.own_index < self.min_child_index:
            # Found the minimum index node.
            self._replace_with(child_node)                        
            child_node._refresh_upwards()
            
        else:
            # For now:
            assert not self.is_terminal()
            
            # Find a minimum index node:
            if self.is_unbalanced():
                self.left_child._add_while_above_capacity(child_node)
            else:
                # For now.
                assert self.is_filled()
                
                if self.left_child._get_min_index_overall() < self.right_child._get_min_index_overall():
                    self.left_child._add_while_above_capacity(child_node)
                else:
                    self.right_child._add_while_above_capacity(child_node)


    def _refresh_upwards(self):
        """A helper method that refreshes the attributes of the nodes enroute to the
        root. This is typically invoked when a new node has been added."""
        
        left_min_child_index_contribution = self.left_child._get_min_index_overall() if self.left_child else np.nan
        right_min_child_index_contribution = self.right_child._get_min_index_overall() if self.right_child else np.nan
        self.min_child_index = np.nanmin([Node.DEFAULT_MIN_CHILD_INDEX, left_min_child_index_contribution, right_min_child_index_contribution])
        
        left_sum_of_child_weights_contribution = self.left_child._get_sum_of_weights_overall() if self.left_child else 0
        right_sum_of_child_weights_contribution = self.right_child._get_sum_of_weights_overall() if self.right_child else 0
        self.sum_of_child_weights = left_sum_of_child_weights_contribution + right_sum_of_child_weights_contribution
        
        left_max_of_child_weights_contribution = self.left_child.get_max_of_weights_overall() if self.left_child else 0
        right_max_of_child_weights_contribution = self.right_child.get_max_of_weights_overall() if self.right_child else 0
        self.max_of_child_weights = max(left_max_of_child_weights_contribution, right_max_of_child_weights_contribution)
                
        left_min_depth_down_contribution = (1 + self.left_child.min_depth_down) if self.left_child else 0
        right_min_depth_down_contribution = (1 + self.right_child.min_depth_down) if self.right_child else 0
        self.min_depth_down = min(left_min_depth_down_contribution, right_min_depth_down_contribution)

        if self.parent:
            self.parent._refresh_upwards()


    def _get_cumulative_sampling_probabilities(self):
        """A helper method that returns the weight-informed sampling 
        probability cut-offs to use for a node."""
        total = self._get_sum_of_weights_overall()
        
        up_to_end_of_own = self.own_weight
        up_to_end_of_left = up_to_end_of_own + (self.left_child._get_sum_of_weights_overall() if self.left_child else 0)
        
        return np.array([up_to_end_of_own, up_to_end_of_left, total]) / total

    
    def _sample_single(self):
        """A helper method that returns a weight-based sample of size 1."""

        temp = self

        while True:

            r = random.random()            
            cumulative_sampling_probabilities = temp._get_cumulative_sampling_probabilities()            
        
            if r <= cumulative_sampling_probabilities[0]:
                return temp
            elif r <= cumulative_sampling_probabilities[1]:
                # For now:
                assert temp.left_child                
                temp = temp.left_child
            else:
                # For now:
                assert temp.right_child
                temp = temp.right_child
                        
        
    def __repr__(self):
        """Returns a user-friendly representation of the node."""

        if Node.VERBOSE:

            out = f'''
                ----- {hex(id(self))} -----

                parent: {hex(id(self.parent)) if self.parent else '-'}
                left_child: {hex(id(self.left_child)) if self.left_child else '-'}
                right_child: {hex(id(self.right_child)) if self.right_child else '-'}

                own_index: {self.own_index}
                min_child_index: {self.min_child_index}

                own_weight: {self.own_weight}
                sum_of_child_weights: {self.sum_of_child_weights}
                max_of_child_weights: {self.max_of_child_weights}

                min_depth_down: {self.min_depth_down}

                previously_been_added: {self.previously_been_added}
                has_been_replaced: {self.has_been_replaced}
            '''

        else:
            out = f"{hex(id(self))}::: P: {hex(id(self.parent)) if self.parent else '-'}; L: {hex(id(self.left_child)) if self.left_child else '-'}; R:{hex(id(self.right_child)) if self.right_child else '-'}; own_index: {self.own_index}; min_child_index: {self.min_child_index}; own_weight:{self.own_weight}; sum_of_child_weights: {self.sum_of_child_weights}; max_of_child_weights: {self.max_of_child_weights}; min_depth_down: {self.min_depth_down}"

        return out
