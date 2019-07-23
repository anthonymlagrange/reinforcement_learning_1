from collections import namedtuple
from model_dueling import QDuelingNetwork
from model_standard import QStandardNetwork
import numpy as np
from prioritized_replay_buffer import PrioritizedReplayBuffer
import random
from replay_buffer import ReplayBuffer
import time
import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """An agent that interacts with and learns from its environment. As its baseline it
    uses Deep Q-Learning (DQN). The following can optionally be enabled in all possible
    combinations:
        * Double DQN (DDQN)
        * Prioritized Experience Replay
        * Dueling Network Architecture
    """


    def __init__(self, 
                 
                 state_size, 
                 action_size, 
                 
                 buffer_size=int(1e5),
                 batch_size=64,
                 gamma=0.99,
                 tau=0.025,
                 learning_rate=5e-4,
                 update_every=4,
                                  
                 enable_double_dqn=False, 
                 enable_prioritized_experience_replay=False, 
                 enable_dueling_network=False,
                 
                 alpha=0.6
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): the dimension of each state
            action_size (int): the dimension of each action
            
            buffer_size (int): the replay buffer size
            batch_size (int): the minibatch size 
            gamma (float): the reward discount factor
            tau (float): for soft updates of the target parameters
            learning_rate (float): the learning rate
            update_every (int): controls how regularly the network learns
            
            enable_double_dqn (bool): enables Double DQN (DDQN)
            enable_prioritized__experience_replay (bool): enables Prioritized Experience Replay
            enable_dueling_network (bool): enables a Dueling Network architecture
            
            alpha (float): the priority dampening effect in Prioritized Experience Replay
        """
        self.state_size = state_size
        self.action_size = action_size
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.update_every = update_every
        
        self.enable_double_dqn = enable_double_dqn
        self.enable_prioritized_experience_replay = enable_prioritized_experience_replay
        self.enable_dueling_network = enable_dueling_network
        
        self.alpha = alpha                
        
        # Instantiate the local and target networks.
        Network = QStandardNetwork if not self.enable_dueling_network else QDuelingNetwork
        self.qnetwork_local = Network(state_size, action_size).to(device)
        self.qnetwork_target = Network(state_size, action_size).to(device)
        # Starting off with the same random weights in self.qnetwork_local and self.qnetwork_target.
        self._perform_soft_update(self.qnetwork_local, self.qnetwork_target, tau=1)
        
        # Instantiate the experience memory.
        if not self.enable_prioritized_experience_replay:
            self.memory = ReplayBuffer(self.buffer_size)
        else:
            self.memory = PrioritizedReplayBuffer(self.buffer_size, alpha=self.alpha)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        
        # Start counting the steps.
        self.t_step = 0
        
        # Clear the performance timers.
        self.reset_timers()
        
        
    def act(self, state, epsilon):
        """Returns an action for the given state and epison-greedy value as per the current policy.
        
        Params
        ======
            state (array_like): the current state
            epsilon (float): the epsilon-greedy value
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))        
                
        
    def step(self, state, action, reward, next_state, done, beta=None):
        """Updates the policy based on the state, action, reward and next_state.
        Also takes into account that the episode might be done. 
        
        For Prioritized Experience Replay, also receives the beta value that should 
        be used for the de-biasing factor.
        
        Params
        ======
            state (array_like): the current state
            action (int): the action taken
            reward (float): the reward received for taking the action in the state
            next_state (array_like): the resulting state
            done (bool): indicates whether the episode is done or not
            beta (float): For Prioritized Experience Replay, the beta value that
                should be used next for the de-biasing factor
        """

        # Save experience in replay memory        
        self.memory.add(state, action, reward, next_state, done)
                            
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self._learn(experiences, beta)

                
    def save(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)

        
    def reset_timers(self):
        """Resets performance timers.        
        """
        self.time_1 = 0
        self.time_2 = 0
        self.time_3 = 0
        self.time_4 = 0                
                
                
    def _learn(self, experiences, beta):
        """Updates Q by learning from a batch of experiences.

        Params
        ======
            experiences (tuple): A batch of sampled experiences.
        """

        predictions, targets, nodes = self._calculate_predictions_and_targets(experiences)
        
        if not self.enable_prioritized_experience_replay:
            self._learn_from_experience_replay(predictions, targets)            
        else:
            self._learn_from_prioritized_experience_replay(predictions, targets, nodes, beta)


    def _calculate_predictions_and_targets(self, experiences):
        """From a batch of sampled experiences, calculates the predictions and targets.
        Also returns the nodes of the samples.
        
        Params
        ======
            experiences (tuple): a batch of sampled experiences
        
        Returns
        =======
            A tuple of predictions, targets and nodes.
        """
        
        in_states, in_actions, in_rewards, in_next_states, in_dones, nodes = experiences
        
        states = torch.from_numpy(in_states).float().to(device)
        actions = torch.from_numpy(in_actions).long().to(device)
        rewards = torch.from_numpy(in_rewards).float().to(device)
        next_states = torch.from_numpy(in_next_states).float().to(device)
        dones = torch.from_numpy(in_dones).float().to(device)
        
        predictions = self.qnetwork_local(states)[torch.range(0, states.shape[0] - 1, dtype=torch.long), torch.squeeze(actions)].to(device)        

        with torch.no_grad():

            if not self.enable_double_dqn:            
                inputs_for_targets = self.qnetwork_target(next_states).to(device)
                targets = (torch.squeeze(rewards) + (1.0 - torch.squeeze(dones)) * self.gamma * inputs_for_targets.max(1)[0]).to(device)
            else:
                temp_1 = self.qnetwork_local(next_states).to(device)
                temp_2 = temp_1.max(1)[1].to(device)
                temp_3 = self.qnetwork_target(next_states)[torch.range(0, next_states.shape[0] - 1, dtype=torch.long), temp_2].to(device)                    
                targets = (torch.squeeze(rewards) + (1.0 - torch.squeeze(dones)) * self.gamma * temp_3).to(device)
        
        return (predictions, targets, nodes)


    def _learn_from_experience_replay(self, predictions, targets):
        """Updates Q by learning from (non-prioritized) Experience Replay.
        
        Params
        ======
            predictions (array_like): batch-size predictions
            targets (array_like): batch-size targets
        """
        
        assert not self.enable_prioritized_experience_replay
        
        td_errors = targets - predictions
        torch.Tensor.clamp_(td_errors, min=-1, max=1)
        
        loss = torch.mean(torch.pow(td_errors, 2))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network:
        self._perform_soft_update(self.qnetwork_local, self.qnetwork_target, tau=self.tau)


    def _learn_from_prioritized_experience_replay(self, predictions, targets, nodes, beta):
        """Updates Q by learning from Prioritized Experience Replay.
        
        Params
        ======
            predictions (array_like): batch-size predictions
            targets (array_like): batch-size targets
            nodes (array_like): the nodes associated with the predictions and targets
            beta (float): The beta value that should be used next for the de-biasing factor        
        """
        
        assert self.enable_prioritized_experience_replay
        
        # Calculate the gradient weights:            
        time_1_start = time.process_time()  
        root = nodes[0].get_root()            
        total_weight = root.get_max_of_weights_overall()  # 'alpha' has already been applied.
        sampled_weights = np.array([n.own_weight for n in nodes])  # 'alpha' has already been applied.
        scaled_weights = sampled_weights / total_weight # P         
        gradient_weights = np.power(self.buffer_size * scaled_weights, -beta)
        gradient_weights = gradient_weights / np.max(gradient_weights)        
        gradient_weights = torch.from_numpy(gradient_weights).float().to(device)
        self.time_1 += time.process_time() - time_1_start  # Measure the performance.        

        # Calculate the TD errors and loss; update the local network weights:
        time_2_start = time.process_time()
        td_errors = targets - predictions
        torch.Tensor.clamp_(td_errors, min=-1, max=1)  # Clip the TD errors for greater stability.        
        loss = torch.mean(torch.pow(td_errors, 2) * gradient_weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # An alternative but less elegant and slower approach involves applying the 
        # gradient weights to the gradient components instead of the loss components
        # as was done above:
        # weighted_gradients = {}
        # for k in range(self.batch_size):
        #   loss_single = self.loss_function(predictions[k], targets[k])  # Where for equivalent self.loss_function is MSE.
        #   self.qnetwork_local.zero_grad()
        #   loss_single.backward(retain_graph=True)                
        #   with torch.no_grad():
        #       for name, param in self.qnetwork_local.named_parameters():
        #       if name not in weighted_gradients:
        #           weighted_gradients[name] = param.grad * gradient_weights[k]
        #       else:
        #           weighted_gradients[name] += param.grad * gradient_weights[k]
        # with torch.no_grad():
        #    for name, param in self.qnetwork_local.named_parameters():      
        #        param.data -= self.learning_rate * weighted_gradients[name]        
        self.time_2 += time.process_time() - time_2_start  # Measure the performance.        
           
        # Update the target network:
        time_3_start = time.process_time()        
        self._perform_soft_update(self.qnetwork_local, self.qnetwork_target, tau=self.tau)                     
        self.time_3 += time.process_time() - time_3_start  # Measure the performance.
                
        # Update the node weights:
        time_4_start = time.process_time()        
        with torch.no_grad():        
            for node, td_error in zip(nodes, td_errors.cpu().numpy()):
                weight = self.memory.calculate_weight_from_raw_weight(td_error)
                node.update_weight(weight)
        self.time_4 += time.process_time() - time_4_start  # Measure the performance.
        

    def _perform_soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau)*target_param.data)
