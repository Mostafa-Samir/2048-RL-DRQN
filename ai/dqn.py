import tensorflow as tf
import numpy as np
import random

from collections import deque

class DQN:

    def __init__(self, qnn,
                       optimizer,
                       session,
                       state_size,
                       actions_count,
                       reply_memory_size=1000,
                       minibatch_size=32,
                       final_exploration_probability=0.05,
                       exploration_period=1000,
                       discount_factor=0.95,
                       summary_writer=None):
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
        state_size: int
            The size of the input states
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
        summary_writer: tf.SummaryWriter
            Tensorflow summary writer
        """


        self.prediction_nn = qnn
        self.target_nn = qnn.clone()
        self.optimizer = optimizer

        self.state_size = state_size
        self.actions_count = actions_count

        self.experience = deque()
        self.reply_memory_size = reply_memory_size
        self.minibatch_size = minibatch_size

        self.epsilon_initial = 1.0
        self.epsilon_final = final_exploration_probability
        self.exploration_period = exploration_period

        self.discount = tf.constant(discount_factor)

        # counters for the Trainer
        self.action_requests = 0  # captures how many times the action method was called
        self.iteration = 0  # captures how many times the train method was called

        self.summary_writer  = summary_writer
        self.session = session

        self.build_graph()


    def build_graph(self):
        """
        Builds Tensorflow computation graph for the DQN trainer
        """

        # placeholder for the inputs
        self.states = tf.placeholder(tf.float32, (None, self.state_size))
        self.next_states = tf.placeholder(tf.float32, (None, self.state_size))
        #self.final_states_filter = tf.placeholder(tf.float32, (None,))
        self.rewards = tf.placeholder(tf.float32, (None,))
        self.experience_action_filter = tf.placeholder(tf.float32, (None, self.actions_count))

        # pi(S) = argmax Q(S,a) over a
        self.actions_scores = tf.identity(self.prediction_nn(self.states))
        self.predicted_actions = tf.argmax(self.actions_scores, dimension=1)

        # future_estimate = R + gamma * max Q(S',a') over a'
        self.next_actions_scores = tf.stop_gradient(self.target_nn(self.next_states))
        self.target_values = tf.reduce_max(self.next_actions_scores, reduction_indices=[1,])
        self.future_estimate = self.rewards * self.discount * self.target_values

        # predicted_value = Q(S, a)
        self.experience_action_score = self.actions_scores * self.experience_action_filter
        self.predicted_value = tf.reduce_sum(self.experience_action_score, reduction_indices=[1,])

        # L = E[(R + gamma * max Q(S',a') - Q(S,a))^2]
        self.loss = tf.reduce_mean(tf.square(self.future_estimate - self.predicted_value))

        # training computations
        gradients = self.optimizer.compute_gradients(self.loss)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, 5), var)

        self.train_computation  = self.optimizer.apply_gradients(gradients)

        # summaries
        # Add histograms for gradients.
        for grad, var in gradients:
            tf.histogram_summary(var.name, var)
            if grad is not None:
                tf.histogram_summary(var.name + '/gradients', grad)
        # loss summary
        tf.scalar_summary("loss", self.loss)

        self.collect_summries = tf.merge_all_summaries()
        self.no_op = tf.no_op()


    def remember(self, state, action, reward, nextstate):
        """
        remembers an experience in the reply memory

        Parameters:
        ----------
        state: numpy.ndarray
            The initial state of the experience
        action: int
            The action took at the intial state
        reward: float
            The reward recived after taking the action
        nextstate: numpy.ndarray
            The next state to which the agent transitioned after
            taking the action
        """

        new_experience = (state, action, reward, nextstate)
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

        state = np.array([state], dtype=np.float32)

        if not play_mode:
            self.action_requests += 1
            epsilon = self.current_epsilon()

            if random.random() < epsilon:
                return np.random.choice(available_actions)
            else:
                actions_scores = self.session.run(self.actions_scores, {self.states: state})
                actions_scores = np.array(actions_scores)[:, available_actions]
                return np.argmax(actions_scores, axis=1)

        else:
            actions_scores = self.session.run(self.actions_scores, {self.states: state})
            actions_scores = np.array(actions_scores)[:, available_actions]
            return np.argmax(actions_scores, axis=1)


    def train(self):
        """
        Runs a training step on the experience memory
        """

        if len(self.experience) < self.minibatch_size:
            return

        random_indecies = random.sample(range(len(self.experience)), self.minibatch_size)
        samples = [self.experience[i] for i in random_indecies]

        states = np.empty((self.minibatch_size, self.state_size))
        chosen_actions_filters = np.zeros((self.minibatch_size, self.actions_count))
        rewards = np.empty((self.minibatch_size,))
        nextstates = np.empty((self.minibatch_size, self.state_size))

        for i, (state, action, reward, nextstate) in enumerate(self.experience):
            states[i] = state
            chosen_actions_filters[i][action] = 1.
            rewards[i] = reward
            nextstates[i] = nextstate

        summarize = self.iteration % 100 == 0 and self.summary_writer is not None

        loss,_,summary = self.session.run([
            self.loss,
            self.train_computation,
            self.collect_summries if summarize else self.no_op
        ], {
            self.states: states,
            self.experience_action_filter: chosen_actions_filters,
            self.rewards: rewards,
            self.next_states: nextstates
        })

        # update the target network
        self.target_nn = self.prediction_nn.clone()

        if summarize:
            self.summary_writer.add_summary(summary, self.iteration)

        self.iteration +=1
