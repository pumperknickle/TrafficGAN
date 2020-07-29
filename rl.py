from models import Generator
import numpy as np


class Agent(object):
    """
    On each step, Agent act on state.
    Then Environment return next state, reward, and so on.
    """

    def __init__(self, B, V, E, H, lr=1e-3):
        """
        # Arguments:
            B: int, batch_size
            V: int, Vocabulary size
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001
        """
        self.num_actions = V
        self.B = B
        self.V = V
        self.E = E
        self.H = H
        self.lr = lr
        self.eps = 0.1
        self.generator = Generator(B, V, E, H, lr)

    def act(self, state, epsilon=0, deterministic=False, stateful=True):
        """
        # Arguments:
            state: numpy array, dtype=int, shape = (B, t)
            epsilon: float, 0 <= epsilon <= 1,
                if epsilon is 1, the Agent will act completely random.
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1)
        """
        token = state[:, -1].reshape([-1, 1])
        return self.act_on_state(token, epsilon=epsilon, stateful=stateful, deterministic=deterministic)

    def act_on_state(self, token, epsilon=0, deterministic=False, stateful=True, PAD=0, EOS=2):
        '''
        # Arguments:
            token: numpy array, dtype=int, shape = (B, 1),
                token indicates current token.
            epsilon: float, 0 <= epsilon <= 1,
                if epsilon is 1, the Agent will act completely random.
        # Returns:
            action: numpy array, dtype=int, shape = (B, 1)
        '''
        is_PAD = token == PAD
        is_EOS = token == EOS
        is_end = is_PAD.astype(np.int) + is_EOS.astype(np.int)
        is_end = 1 - is_end
        is_end = is_end.reshape([self.B, 1])
        if stateful:
            probs = self.generator.predict(token, stateful=True)
        else:
            probs, _, _ = self.generator.predict(token, stateful=False)
        if np.random.rand() <= epsilon:
            action = np.random.randint(low=0, high=self.num_actions, size=(self.B, 1))
        elif not deterministic:
            action = self.generator.sample_token(probs).reshape([self.B, 1])
        else:
            action = np.argmax(probs, axis=-1).reshape([self.B, 1])
        return action * is_end

    """
    Sets the lstm hidden state to 0s
    """
    def reset(self):
        self.generator.reset_rnn_state()

    def save(self, path):
        self.generator.save(path)

    def load(self, path):
        self.generator.load(path)


class Environment(object):
    """
    Environment is responsible for managing generators and computing the state action values or rewards
    """
    def __init__(self, discriminator, B, T, g_beta, BOS=1, n_sample=16):
        """
        Environment class for Reinforced Learning
        # Arguments:
            discriminator: keras model
            B: Batch Size
            T: Max Time Steps
            g_beta: SeqGAN.rl.Agent, copy of Agent
                params of g_beta.generator should be updated with those of original
                generator on regular occasions.
        # Optional Arguments
            n_sample: int, default is 16, the number of Monte Calro search sample
        """
        self.B = B
        self.T = T
        self.n_sample = n_sample
        self.BOS = BOS
        self.discriminator = discriminator
        self.g_beta = g_beta
        self.t = 1
        self._state = np.zeros([self.B, 1], dtype=np.int32)
        self.reset()


    def get_state(self):
        return self._state

    def reset(self):
        self.t = 1
        self._state = np.zeros([self.B, 1], dtype=np.int32)
        self._state[:, 0] = self.BOS
        self.g_beta.reset()

    def _append_state(self, word, state=None):
        """
        # Arguments:
            word: numpy array, dtype=int, shape = (B, 1)
        """
        word = word.reshape(-1, 1)
        if state is None:
            self._state = np.concatenate([self._state, word], axis=-1)
        else:
            return np.concatenate([state, word], axis=-1)

    def step(self, action):
        """
        Step t -> t + 1 and returns a result of the Agent action.
        # Arguments:
            action: numpy array, dtype=int, shape = (B, 1),
                state is Y_0:t-1, and action is y_t
        # Returns:
            next_state: numpy array, dtype=int, shape = (B, t)
            reward: numpy array, dtype=float, shape = (B, 1)
            is_episode_end: bool
            info: dict
        """
        self.t = self.t + 1

        reward = self.Q(action, self.n_sample)
        is_episode_end = self.t >= self.T

        self._append_state(action)
        next_state = self._state
        info = None

        return [next_state, reward, is_episode_end, info]

    def Q(self, action, n_sample=16):
        """
        State-Action value function using Rollout policy
        # Arguments:
            action: numpy array, dtype=int, shape = (B, 1)
        # Optional Arguments:
            n_sample: int, default is 16, number of samples for Monte Calro Search
        # Returns:
            reward: numpy array, dtype=float, shape = (B, ), State-Action value
        # Requires:
            t, T: used to define time range.
            state: determined texts, Y[0:t-1], used for Rollout.
            action: next words, y[t], used for sentence Y[0:t].
            g_beta: Rollout policy.
        """
        reward = np.zeros([self.B, 1])
        Y_base = self._state  # Initial case
        self.g_beta.act(Y_base, epsilon=self.g_beta.eps)
        Y_base = self._append_state(action, state=Y_base)
        # LSTM Hidden States after action is applied
        next_h, next_c = self.g_beta.generator.get_rnn_state()
        if self.t >= self.T:
            return self.discriminator.predict(Y_base)
        # Rollout
        for idx_sample in range(n_sample):
            Y = Y_base
            self.g_beta.generator.set_rnn_state(next_h, next_c)
            for _ in range(self.t, self.T):
                y_tau = self.g_beta.act(Y, epsilon=self.g_beta.eps)
                Y = self._append_state(y_tau, state=Y)
            reward += self.discriminator.predict(Y) / n_sample
        self.g_beta.generator.set_rnn_state(next_h, next_c)
        return reward








