import numpy as np
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Dropout
from keras.layers import Dense, Embedding, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.utils import to_categorical
import pickle
import tensorflow as tf


def GeneratorPretraining(V, E, H):
    '''
    Model for Generator pretraining. This model's weights should be shared with
        Generator once pretraining is complete
    # Arguments:
        V: int, Vocabulary size
        E: int, Embedding size
        H: int, LSTM hidden size
    # Returns:
        generator_pretraining: keras Model
            input: word ids, shape = (B, T)
            output: word probability, shape = (B, T, V)
    '''
    input = Input(shape=(None,), dtype='int32', name='Input')  # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input)  # (B, T, E)
    out = LSTM(H, return_sequences=True, name='LSTM')(out)  # (B, T, H)
    out = TimeDistributed(Dense(V, activation='softmax', name='DenseSoftmax'), name='TimeDenseSoftmax')(out)  # (B, T, V)
    generator_pretraining = Model(input, out)
    return generator_pretraining


class Generator:
    def __init__(self, B, V, E, H, lr=1e-3):
        """
        # Arguments:
            B: int, Batch size
            V: int, Vocabulary size
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001
        """
        self.B = B
        self.V = V
        self.E = E
        self.H = H
        self.lr = lr
        self.packet_size_policy, self.packet_size_predictor, self.extract_hidden_state = self.build_packet_size_graph()
        self.h = np.zeros([self.B, self.H])
        self.c = np.zeros([self.B, self.H])

    def build_packet_size_graph(self):
        # Inputs
        state_in = Input(shape=[1])
        reward = Input(shape=[1])
        h_in = Input(shape=(self.H,))
        c_in = Input(shape=(self.H,))

        # Embedding
        embedding = Embedding(self.V, self.E, mask_zero=True)
        embeddingOutput = embedding(state_in)  # (B, 1, E)

        # LSTM
        lstm = LSTM(self.H, return_state=True)
        lstm_output, next_h, next_c = lstm(embeddingOutput, initial_state=[h_in, c_in])

        # Dense SoftMax
        dense = Dense(self.V, activation='softmax')
        probOut = dense(lstm_output)  # (B, V)

        # Negative log likelihood
        def packet_loss(y_true, y_pred):
            clipped_y_pred = K.clip(y_pred, 1e-8, 1 - 1e-8)
            log_lik = y_true * K.log(clipped_y_pred)
            return K.sum(-log_lik * reward)

        extract_hidden_state = Model(inputs=[state_in, h_in, c_in], outputs=[next_h, next_c])
        packet_size_policy = Model(inputs=[state_in, h_in, c_in, reward], outputs=[probOut])
        packet_size_policy.compile(optimizer=Adam(lr=self.lr), loss=packet_loss)
        packet_size_predictor = Model(inputs=[state_in, h_in, c_in], outputs=[probOut])
        return packet_size_policy, packet_size_predictor, extract_hidden_state

    def reset_rnn_state(self):
        self.h = np.zeros([self.B, self.H])
        self.c = np.zeros([self.B, self.H])

    def set_rnn_state(self, h, c):
        '''
        # Arguments:
            h: np.array, shape = (B,H)
            c: np.array, shape = (B,H)
        '''
        self.h = h
        self.c = c

    def get_rnn_state(self):
        return self.h, self.c

    def predict(self, state, stateful=True):
        """
        Predict next action(word) probability
        # Arguments:
            state: np.array, previous word ids, shape = (B, 1)
        # Optional Arguments:
            stateful: bool, default is True
                if True, update lstm(h, c) to Generator.h, Generator.c
                    and return prob.
                else, return prob, next_h, next_c without updating states.
        # Returns:
            prob: np.array, shape=(B, V)
        """
        next_h, next_c = self.extract_hidden_state.predict([state, self.h, self.c])
        prob = self.packet_size_predictor.predict([state, self.h, self.c])
        if stateful:
            self.h = next_h
            self.c = next_c
            return prob
        else:
            return prob, next_h, next_c

    def update(self, state, action, reward, h=None, c=None, stateful=True):
        """
        Update weights by Policy Gradient.
        # Arguments:
            state: np.array, Environment state, shape = (B, 1) or (B, t)
                if shape is (B, t), state[:, -1] will be used.
            action: np.array, Agent action, shape = (B, )
                In training, action will be converted to onehot vector.
                (Onehot shape will be (B, V))
            reward: np.array, reward by Environment, shape = (B, )
        # Optional Arguments:
            h: np.array, shape = (B, H), default is None.
                if None, h will be Generator.h
            c: np.array, shape = (B, H), default is None.
                if None, c will be Generator.c
            stateful: bool, default is True
                if True, update rnn_state(h, c) to Generator.h, Generator.c
                    and return loss.
                else, return loss, next_h, next_c without updating states.
        # Returns:
            loss: np.array, shape = (B, )
            next_h: (if stateful is True)
            next_c: (if stateful is True)
        """
        if h is None:
            h = self.h
        if c is None:
            c = self.c
        state = state[:, -1].reshape(-1, 1)
        reward = reward.reshape(-1, 1)
        next_h, next_c = self.extract_hidden_state.predict([state, h, c])
        Y = np.array(to_categorical(action, self.V))
        history = self.packet_size_policy.fit([np.array(state), np.array(h), np.array(c), np.array(reward)], Y, batch_size=self.B)
        loss = history.history["loss"][-1]
        if stateful:
            self.h = next_h
            self.c = next_c
            return loss
        else:
            return loss, next_h, next_c

    def sample_token(self, prob):
        '''
        # Arguments:
            prob: numpy array, dtype=float, shape = (B, V),
        # Returns:
            action: numpy array, dtype=int, shape = (B, )
        '''
        action = np.zeros((self.B,), dtype=np.int32)
        for i in range(self.B):
            p = prob[i]
            action[i] = np.random.choice(self.V, p=p)
        return action

    def sample_sequence(self, T, PAD=0, BOS=1, EOS=2):
        '''
        # Arguments:
            T: int, max time steps
        # Optional Arguments:
            BOS: int, id for Begin Of Sequence
        # Returns:
            actions: numpy array, dtype=int, shape = (B, T)
        '''
        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)
        action[:, 0] = BOS
        actions = action
        for _ in range(T - 1):
            token = action[:, -1].reshape([-1, 1])
            is_PAD = token == PAD
            is_EOS = token == EOS
            is_end = is_PAD.astype(np.int) + is_EOS.astype(np.int)
            is_end = 1 - is_end
            is_end = is_end.reshape([self.B, 1])
            prob = self.predict(action)
            action = self.sample_token(prob).reshape([self.B, 1])
            action = action * is_end
            actions = np.concatenate([actions, action], axis=-1)
        self.reset_rnn_state()
        return actions

    def generate_samples(self, T, num):
        """
        Generate sample sequences
        # Arguments:
            T: int, max time steps
            num: int, number of sentences
        """
        samples = []
        for _ in range(num // self.B + 1):
            actions = self.sample_sequence(T)
            actions_list = actions.tolist()
            samples += actions_list
        return samples


    def save(self, path):
        weights = self.packet_size_policy.get_weights()
        with open(path, 'wb') as f:
            pickle.dump(weights, f)


    def load(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        self.packet_size_policy.set_weights(weights)


def SignatureDiscriminator(S, H=64):
    """
    Discriminator model using signature analysis.
    # Arguments:
        H: int, hidden size
        S: int, signature cardinality (# of total signatures)
    # Returns:
        discriminator: keras model
            input: floats, shape = (B, S) (S is total number of signatures)
            output: probability of true data or not, shape = (B, 1)
    """
    model = Sequential()
    model.add(Dense(H, activation='relu', input_shape=(S,)))
    model.add(Dense(H, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def Discriminator(V, E, H=64, dropout=0.1):
    """
    Discriminator model.
    # Arguments:
        V: int, Vocabulary size
        E: int, Embedding size
        H: int, LSTM hidden size
        dropout: float
    # Returns:
        discriminator: keras model
            input: word ids, shape = (B, T)
            output: probability of true data or not, shape = (B, 1)
    """
    inp = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(inp)  # (B, T, E)
    out = LSTM(H)(out)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = Model(inp, out)
    return discriminator


def Highway(x, num_layers=1, activation='relu', name_prefix=''):
    '''
    Layer wrapper function for Highway network
    # Arguments:
        x: tensor, shape = (B, input_size)
    # Optional Arguments:
        num_layers: int, dafault is 1, the number of Highway network layers
        activation: keras activation, default is 'relu'
        name_prefix: str, default is '', layer name prefix
    # Returns:
        out: tensor, shape = (B, input_size)
    '''
    input_size = K.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio_name = '{}Highway/Gate_ratio_{}'.format(name_prefix, i)
        fc_name = '{}Highway/FC_{}'.format(name_prefix, i)
        gate_name = '{}Highway/Gate_{}'.format(name_prefix, i)

        gate_ratio = Dense(input_size, activation='sigmoid', name=gate_ratio_name)(x)
        fc = Dense(input_size, activation=activation, name=fc_name)(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]), name=gate_name)([fc, x, gate_ratio])
    return x
