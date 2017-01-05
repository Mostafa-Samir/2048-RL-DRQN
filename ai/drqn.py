import tensorflow as tf
import numpy as np
import random
from collections import deque

class DRQN:

    def __init__(self,
        qnn,
        optimizer,
        session,
        state_shape,
        actions_count,
        reply_memory_size=100,
        minibatch_size=32,
        final_exploration_probability=0.05,
        exploration_period=1000,
        discount_factor=0.95,
        unrollings_num=16,
        target_freeze_period=1000,
    ):

        """
        constructs a deep recurrent q-learning trainer, as described at
        https://arxiv.org/pdf/1507.06527v3.pdf

        Parameters:
        ----------
        qnn: ai.nn.NeuralNetwork
            The deep neural network approximating the q-value function
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
            reward discount factor
        unrollings_num: int
            the number of steps back in time to train the RNN on
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
        self.current_episode = []
        self.unrollings_num = unrollings_num
        self.minibatch_size = minibatch_size

        self.epsilon_initial = 1.0
        self.epsilon_final = final_exploration_probability
        self.exploration_period = exploration_period
        self.discount = discount_factor
        self.freeze_period = target_freeze_period

        self.action_requests = 0
        self.iteration = 0

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
        with self.session.graph.as_default():
            min_values = tf.reduce_min(input_tensor, reduction_indices, keep_dims=True)
            not_c = tf.abs(c - 1)

            return tf.reduce_max(input_tensor * c + not_c * min_values, reduction_indices)

    def _argmax(self, input_tensor, dimension, c):
        """
        a constrainable version of tf.argmax

        Parameters:
        -----------
        input_tensor: Tensor
        dimension: Tensor
        c: Tensor
            The constraints tensor
            A tensor of 0s and 1s where 1s represent the elements the reduction
            should be made on, and 0s represent discarded elements
        """
        with self.session.graph.as_default():
            min_values = tf.reduce_min(input_tensor, reduction_indices=[dimension,], keep_dims=True)
            not_c = tf.abs(c - 1)

            return tf.argmax(input_tensor * c + not_c * min_values, dimension)

    def build_graph(self):
        """
        builds the computation graph of the training
        """
        persumed_state_shape = tuple([None, None] + self.state_shape)

        # placeholders
        self.states = tf.placeholder(tf.float32, persumed_state_shape)
        self.next_states = tf.placeholder(tf.float32, persumed_state_shape)
        self.rewards = tf.placeholder(tf.float32, (None,))
        self.transition_action_filters = tf.placeholder(tf.float32, (None, self.actions_count))
        self.next_legal_actions_filters = tf.placeholder(tf.float32, (None, self.actions_count))
        self.query_actions_filter = tf.placeholder(tf.float32, (None, self.actions_count))

        next_actions_scores = tf.stop_gradient(self.target_nn(self.next_states))
        target_values = self._reduce_max(next_actions_scores, reduction_indices=[1,], c=self.next_legal_actions_filters)
        future_estimate = self.rewards + self.discount * target_values

        actions_scores = tf.identity(self.prediction_nn(self.states))
        transition_action_score = actions_scores * self.transition_action_filters
        predicted_value = tf.reduce_sum(transition_action_score, reduction_indices=[1, ])

        self.loss = tf.reduce_mean(tf.square(future_estimate - predicted_value), name='loss')
        gradients = self.optimizer.compute_gradients(self.loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_value(grad, -1, 1), var)

        self.finalize = self.optimizer.apply_gradients(gradients)

        # node for actions query
        self.query_action = self._argmax(actions_scores, dimension=1, c=self.query_actions_filter)


    def train(self):
        """
        tarins the prediction network on a randomly sampled episode
        """
        if len(self.experience) < self.minibatch_size:
            return

        # sample a minibatch_size of random episode with a number of transitions >= unrollings_num
        random_episodes_indecies = np.random.choice(len(self.experience), self.minibatch_size)
        random_episodes = []
        for index in random_episodes_indecies:
            episode = self.experience[index]

            # 0:random_transitions_space is the range from which a random transition
            # can be picked up while having unrollings_num - 1 transitions after it
            random_transitions_space = len(episode) - self.unrollings_num
            random_start = np.random.choice(random_transitions_space, 1)

            random_episodes.append(episode[random_start:random_start + self.unrollings_num])

        state_shape = tuple([self.minibatch_size, self.unrollings_num] + self.state_shape)

        # prepare the training data
        states = np.empty(state_shape, dtype=np.float32)
        next_states = np.empty(state_shape, dtype=np.float32)
        rewards = np.empty((self.minibatch_size, self.unrollings_num, ), dtype=np.float32)
        transition_action_filters = np.zeros((self.minibatch_size, self.unrollings_num, self.actions_count), dtype=np.float32)
        next_legal_actions_filters = np.zeros((self.minibatch_size, self.unrollings_num, self.actions_count), dtype=np.float32)

        for i, episode in enumerate(random_episodes):
            for j, transition in enumerate(episode):
                state, action, reward, nextstate, next_legal_actions = transition

                states[i,j], rewards[i,j], next_states[i,j] = state, reward, nextstate
                transition_action_filters[i,j][action] = 1.0
                next_legal_actions_filters[i,j][next_legal_actions] = 1.0

        self.prediction_nn.clearLSTMS(self.session)
        self.target_nn.clearLSTMS(self.session)

        loss,_ = self.session.run([self.loss, self.finalize], {
            self.states: states,
            self.next_states: next_states,
            self.rewards: np.reshape(rewards, (self.minibatch_size * self.unrollings_num, )),
            self.transition_action_filters: np.reshape(transition_action_filters, (self.minibatch_size * self.unrollings_num, self.actions_count)),
            self.next_legal_actions_filters: np.reshape(next_legal_actions_filters, (self.minibatch_size * self.unrollings_num, self.actions_count))
        })

        if self.iteration != 0 and self.iteration % self.freeze_period == 0:
            self.target_nn.assign_to(self.prediction_nn, self.session)

        self.iteration += 1

        return loss, self.iteration

    def remember(self, state, action, reward, nextstate, next_legal_actions, last_transition=False):
        """
        remembers a transition in the reply memory

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
        last_transition: bool
            a flag indicating whether this is the last transition in an episode
        """

        new_transition = (state, action, reward, nextstate, next_legal_actions)
        self.current_episode.append(new_transition)

        if last_transition:
            if len(self.current_episode) >= self.unrollings_num:
                # only record episodes with at least unrollings_num transitions
                self.experience.append(list(self.current_episode))
            self.current_episode = []
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
        epsilon = self.current_epsilon()
        will_explore = np.random.random_sample() < epsilon

        if not play_mode and will_explore:
            return np.random.choice(available_actions)
        else:
            query_state = np.reshape(state, tuple([1,1] + self.state_shape)).astype(np.float32)
            query_actions_filter = np.zeros((1, self.actions_count), dtype=np.float32)
            query_actions_filter[0][available_actions] = 1.0

            chosen_action = self.session.run(self.query_action, {
                self.states: query_state,
                self.query_actions_filter: query_actions_filter
            })

            return chosen_action[0]


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
        for episode in checkpoint_data['experience']:
            ep = []
            for transition in episode:
                ep.append(tuple(transition))

            self.experience.append(ep)
