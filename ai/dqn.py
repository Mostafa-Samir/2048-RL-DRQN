import tensorflow as tf
import numpy as np
import random

from collections import deque

class DQN:

    def __init__(self, qnn,
                       optimizer,
                       session,
                       state_shape,
                       actions_count,
                       reply_memory_size=10000,
                       minibatch_size=32,
                       final_exploration_probability=0.05,
                       exploration_period=1000,
                       discount_factor=1.0,
                       target_freeze_period=1000):
        """
        constructs a DQN Trainer
        An implemntation of:
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        Parameters:
        ----------
        qnn: DFCNN
            The neural network representation of Q-value function
        optimizer: tf.solver.*
            Tensorflow solver to train the network with
        session: tf.Session
            Tensorflow session to run the operations in
        state_shape: list
            The shape of the single state tensor
        actions_count: int
            The maximum number of actions that could be carried out
        reply_memory_size: int
            The size of the experience reply memory
        minibatch_size: int
            The size of the random sample of the experience
            to train the network on each step
        final_exploration_probability: float [0, 1]
            The final value of the exploration probability that the model
            will reach after decreasing from 1.0 over time
        exploration_period: int
            The number of steps in which the exploration probability will
            decrease from 1.0 to final_exploration_probability
        discount_factor: float [0, 1]
            The MDP's reward discount factor
        target_freeze_period: int
            The number of steps the target qnn is kept frozen
        """


        self.prediction_nn = qnn
        self.target_nn = qnn.clone()
        self.optimizer = optimizer

        self.state_shape = state_shape
        self.actions_count = actions_count

        self.experience = deque()
        self.reply_memory_size = reply_memory_size
        self.minibatch_size = minibatch_size

        self.epsilon_initial = 1.0
        self.epsilon_final = final_exploration_probability
        self.exploration_period = exploration_period

        self.discount = tf.constant(discount_factor)

        self.freeze_period = target_freeze_period

        # counters for the Trainer
        self.action_requests = 0  # captures how many times the action method was called
        self.iteration = 0  # captures how many times the train method was called

        self.session = session

        self.build_graph()

    def _reduce_max(self, input_tensor, reduction_indices, c):
        """
        a constrainable version of tf.reduce_max

        Parameters:
        -----------
        input_tensor: Tensor
        reduction_indices: Tensor
        c: Tensor
            The constraints tensor
            A tensor of 0s and 1s where 1s represent the elements the reduction
            should be made on, and 0s represent discarded elements
        """

        min_values = tf.reduce_min(input_tensor, reduction_indices, keep_dims=True)
        not_c = tf.abs(c - 1)

        return tf.reduce_max(input_tensor * c + not_c * min_values, reduction_indices)


    def build_graph(self):
        """
        Builds Tensorflow computation graph for the DQN trainer
        """

        persumed_state_shape = tuple([None] + self.state_shape)

        # placeholder for the inputs
        self.states = tf.placeholder(tf.float32, persumed_state_shape)
        self.next_states = tf.placeholder(tf.float32, persumed_state_shape)
        self.rewards = tf.placeholder(tf.float32, (None,))
        self.experience_action_filter = tf.placeholder(tf.float32, (None, self.actions_count))
        self.next_legal_actions_filter = tf.placeholder(tf.float32, (None, self.actions_count))

        # pi(S) = argmax Q(S,a) over a
        self.actions_scores = tf.identity(self.prediction_nn(self.states))
        self.predicted_actions = tf.argmax(self.actions_scores, dimension=1)

        # future_estimate = R + gamma * max Q(S',a') over a'
        self.next_actions_scores = tf.stop_gradient(self.target_nn(self.next_states))
        self.target_values = self._reduce_max(self.next_actions_scores, reduction_indices=[1,], c=self.next_legal_actions_filter)
        self.future_estimate = self.rewards + self.discount * self.target_values

        # predicted_value = Q(S, a)
        self.experience_action_score = self.actions_scores * self.experience_action_filter
        self.predicted_value = tf.reduce_sum(self.experience_action_score, reduction_indices=[1,])

        # L = E[(R + gamma * max Q(S',a') - Q(S,a))^2]
        self.loss = tf.reduce_mean(tf.square(self.future_estimate - self.predicted_value))

        # training computations
        gradients = self.optimizer.compute_gradients(self.loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_value(grad, -1, 1), var)

        self.train_computation  = self.optimizer.apply_gradients(gradients)


    def remember(self, state, action, reward, nextstate, next_legal_actions):
        """
        remembers an experience in the reply memory

        Parameters:
        ----------
        state: list
            The initial state of the experience
        action: int
            The action took at the intial state
        reward: float
            The reward recived after taking the action
        nextstate: list
            The next state to which the agent transitioned after
            taking the action
        next_legal_actions: list
            The list of legal actions for nextstate
        """
        #state = np.reshape(state, self.state_shape).astype(np.float32)
        #nextstate = np.reshape(nextstate, self.state_shape).astype(np.float32)

        new_experience = (state, action, reward, nextstate, next_legal_actions)
        self.experience.append(new_experience)

        if len(self.experience) > self.reply_memory_size:
            self.experience.popleft()


    def current_epsilon(self):
        """
        Computes the value of the epsilon (exploration probability)
        at the calling time step
        """
        t = self.action_requests
        T = self.exploration_period
        if(t >= T):
            return self.epsilon_final

        epsilon0 = self.epsilon_initial
        epsilonT = self.epsilon_final

        return epsilon0 - (t * (epsilon0 - epsilonT)) / T

    def constrained_argmax(self, array, constraints):
        """
        Gets the action with the maximum value provided that it's available
        array: numpy.ndarray (1 x actions_count)
        constraints: list
        """
        max_value = float("-inf")
        max_action = -1
        for i, value in enumerate(array[0]):
            if value >= max_value and i in constraints:
                max_value = value
                max_action = i
        return max_action

    def get_action(self, state, available_actions, play_mode=False):
        """
        Returns the action to be carried out at this state

        Parameters:
        ----------
        state: list
            The state to get the action at
        available_actions: list
            The list of available actions that could be taken at this
            state
        play_mode: bool
            a flag to indicate if the action is needed for play mode
            not for training, this would stop the epsilon-greedy behavior
            and directly use the Qnn
        """

        state = np.reshape(state, tuple([1] + self.state_shape))
        feed_dict = {self.states: state}

        if not play_mode:
            self.action_requests += 1
            epsilon = self.current_epsilon()

            if random.random() < epsilon:
                return np.random.choice(available_actions)
            else:
                actions_scores = self.session.run(self.actions_scores, feed_dict)
                return self.constrained_argmax(actions_scores, available_actions)

        else:
            actions_scores = self.session.run(self.actions_scores, feed_dict)
            return self.constrained_argmax(actions_scores, available_actions)


    def train(self):
        """
        Runs a training step on the experience memory
        """

        if len(self.experience) < self.minibatch_size:
            return

        random_indecies = random.sample(range(len(self.experience)), self.minibatch_size)
        samples = [self.experience[i] for i in random_indecies]

        real_state_shape = tuple([self.minibatch_size] + self.state_shape)

        states = np.empty(real_state_shape, dtype=np.float32)
        chosen_actions_filters = np.zeros((self.minibatch_size, self.actions_count))
        rewards = np.empty((self.minibatch_size,))
        nextstates = np.empty(real_state_shape, dtype=np.float32)
        next_legal_actions_filters = np.zeros((self.minibatch_size, self.actions_count))

        for i, (state, action, reward, nextstate, next_legal_actions) in enumerate(samples):
            states[i] = state
            chosen_actions_filters[i][action] = 1.
            rewards[i] = reward
            nextstates[i] = nextstate

            for action in range(self.actions_count):
                if action not in next_legal_actions:
                    next_legal_actions_filters[i][action] = 1.

        loss,_ = self.session.run([
            self.loss,
            self.train_computation,
        ], {
            self.states: states,
            self.experience_action_filter: chosen_actions_filters,
            self.rewards: rewards,
            self.next_states: nextstates,
            self.next_legal_actions_filter: next_legal_actions_filters,
        })

        if self.iteration != 0 and self.iteration % self.freeze_period == 0:
            self.target_nn.assign_to(self.prediction_nn, self.session)

        self.iteration +=1

        return (loss, self.iteration)


    def serialize(self):
        """
        serializes the inetrnal variables for checkpoint saving

        Returns: dict
        """

        return {
            'experience': list(self.experience),
            'iteration': self.iteration,
            'action_requests': self.action_requests
        }

    def restore(self, checkpoint_data):
        """
        restores internal variables values from a saved checkpoint

        Parameters:
        ----------
        checkpoint_data: dict
        """

        self.iteration = checkpoint_data['iteration']
        self.action_requests = checkpoint_data['action_requests']

        self.experience = deque()
        for transition in checkpoint_data['experience']:
            self.experience.append(tuple(transition))
