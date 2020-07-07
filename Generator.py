from keras.layers.core import Reshape
from keras.layers import Dense, Input, Embedding, Bidirectional, Lambda
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
from keras_self_attention import SeqSelfAttention
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import math
from pcaputilities import save_sequence

max_packet_size = 5000
total_actions = (max_packet_size * 2) + 1


# Decode action value into packet size and direction
def is_start_token(action_value):
    return action_value == 0


def get_packet_size(action_value):
    return abs(action_value - max_packet_size - 1)


def is_direction_in(action_value):
    return action_value - max_packet_size - 1 > 0


class Generator(object):
    def __init__(self, discriminator, segment_length=200, monte_carlo_n=32, epochs=500, alpha=0.0005, n_actions=total_actions, lstm_units=100, dense_units=100,
                 embedding_dim=100,
                 packet_size_policy=None, packet_size_predictor=None, duration_policy=None, duration_predictor=None, psp_fn='pspfn.h5', pspr_fn='pspfrn.h5', dp_fn='dpfn.h5', dpr_fn='dprfn.h5', generated_packets_file='generated_packets.txt', generated_durations_file='generated_durations.txt'):
        self.monte_carlo_n = monte_carlo_n
        self.segment_length = segment_length
        # Discriminator provides "advantages" used in loss function
        self.discriminator = discriminator
        # Learning rate for generator
        self.lr = alpha
        # total number of actions = max length of packet * 2 + 1
        # Encodes direction, length of packet, and start token
        self.n_actions = n_actions
        # number of lstm units
        self.lstm_units = lstm_units
        # number of dense in hidden layer
        self.dense_units = dense_units
        # number of embedding outputs
        self.embedding_dim = embedding_dim
        # previous sequence of actions performed (packets emitted)
        self.action_memory = []
        # previous rewards
        self.reward_memory = []
        # potential packet size choices
        self.action_space = [i for i in range(n_actions)]
        # iterations to train
        self.epochs = epochs
        self.psp_fn = psp_fn
        self.pspr_fn = pspr_fn
        self.dp_fn = dp_fn
        self.dpr_fn = dpr_fn
        self.generated_packet_file = generated_packets_file
        self.generated_durations_file = generated_durations_file
        if packet_size_policy is None or packet_size_predictor is None or duration_policy is None or duration_predictor is None:
            self.packet_size_policy, self.packet_size_predictor = self.build_packet_size_policy_network()
            self.duration_policy, self.duration_predictor = self.build_duration_policy_network()
        else:
            self.packet_size_policy = packet_size_policy
            self.packet_size_predictor = packet_size_predictor
            self.duration_policy = duration_policy
            self.duration_predictor = duration_predictor

    # Define the network model that generates packet sizes and direction
    def build_packet_size_policy_network(self):
        advantages = Input(shape=[1])
        packets_input = Input(shape=(None,))
        packet_embedding = Embedding(input_dim=self.n_actions, output_dim=self.embedding_dim)(packets_input)
        durations_input = Input(shape=(None, 1))
        merged = concatenate([packet_embedding, durations_input])
        lstm1 = LSTM(self.lstm_units, return_sequences=True)(merged)
        attention = SeqSelfAttention(attention_activation='sigmoid')(lstm1)
        lstm2 = LSTM(self.lstm_units)(attention)
        hidden_dense = Dense(self.dense_units, activation='relu')(lstm2)
        packets_output = Dense(self.n_actions, activation='softmax')(hidden_dense)

        # Negative log likelihood
        def packet_loss(y_true, y_pred):
            clipped_y_pred = K.clip(y_pred, 1e-8, 1 - 1e-8)
            log_lik = y_true * K.log(clipped_y_pred)
            return K.sum(-log_lik * advantages)

        packet_size_policy = Model(inputs=[packets_input, durations_input, advantages], outputs=[packets_output])
        packet_size_policy.compile(optimizer=Adam(lr=self.lr), loss=packet_loss)

        packet_size_predictor = Model(inputs=[packets_input, durations_input], outputs=[packets_output])

        return packet_size_policy, packet_size_predictor

    def build_duration_policy_network(self):
        advantages = Input(shape=[1])
        packets_input = Input(shape=(None,))
        packet_embedding = Embedding(input_dim=self.n_actions, output_dim=self.embedding_dim)(packets_input)
        durations_input = Input(shape=(None, 1))
        merged = concatenate([packet_embedding, durations_input])
        lstm1 = LSTM(self.lstm_units, return_sequences=True)(merged)
        attention = SeqSelfAttention(attention_activation='sigmoid')(lstm1)
        lstm2 = LSTM(self.lstm_units)(attention)
        hidden_dense = Dense(self.dense_units, activation='relu')(lstm2)
        duration_output = Dense(2, activation='relu', name="d_output")(hidden_dense)

        # Negative log likelihood gaussian * advantages
        def duration_loss(y_true, y_pred):
            n_dims = int(int(y_pred.shape[1]) / 2)
            mu = y_pred[:, 0:n_dims]
            logsigma = y_pred[:, n_dims:]

            mse = -0.5 * K.sum(K.square((y_true - mu) / K.exp(logsigma)), axis=1)
            sigma_trace = -K.sum(logsigma, axis=1)
            log2pi = -0.5 * n_dims * np.log(2 * np.pi)

            log_likelihood = mse + sigma_trace + log2pi

            return K.mean(-log_likelihood*advantages)

        duration_policy = Model(inputs=[packets_input, durations_input, advantages], outputs=duration_output)
        duration_policy.compile(optimizer=Adam(lr=self.lr), loss=duration_loss)

        duration_predictor = Model(inputs=[packets_input, durations_input], outputs=duration_output)

        return duration_policy, duration_predictor

    def calculate_reward(self, seq):
        states = [seq] * self.monte_carlo_n
        for i in range(self.segment_length - len(seq)):
            actions = self.choose_actions(states)
            for j in range(len(states)):
                states[j] = states[j] + [actions[j]]
        return np.mean(self.discriminator.predict(states))

    def calculate_rewards(self):
        rewards = []
        for i in range(len(self.action_memory)):
            input = self.action_memory[0:i+1]
            rewards.append(self.calculate_reward(input))
        self.reward_memory = rewards

    def choose_actions(self, states):
        action_values = []
        duration_values = []
        for state in states:
            action_values.append([0] + [item[0] for item in state])
            duration_values.append([[0.0]] + [[item[1]] for item in state])
        packet_size_predictor_output = self.packet_size_predictor.predict([np.array(action_values), np.array(duration_values)])
        packet_values = []
        for packet_probs in packet_size_predictor_output:
            packet_values.append(np.random.choice(self.action_space, p=packet_probs))
        extended_action_values = []
        for i in range(len(action_values)):
            action_value = action_values[i][1:]
            packet_value = packet_values[i]
            extended_action_values.append(action_value + [packet_value])
        duration_predictor_output = self.duration_predictor.predict([np.array(extended_action_values), np.array(duration_values)])
        duration_values = []
        for duration_output in duration_predictor_output:
            mean = duration_output[0]
            stddev = math.exp(duration_output[1])
            duration_values.append(math.fabs(np.random.normal(mean, stddev)))
        all_actions = []
        for i in range(len(packet_values)):
            all_actions.append([packet_values[i], duration_values[i]])
        return all_actions

    def choose_action(self, state):
        base_action_values = [item[0] for item in state]
        action_values = np.array([[0] + base_action_values])
        duration_values = np.array([[[0.0]] + [[item[1]] for item in state]])
        packet_size_predictor_output = self.packet_size_predictor.predict([action_values, duration_values])
        packet_probabilities = packet_size_predictor_output[0]
        packet_value = np.random.choice(self.action_space, p=packet_probabilities)
        extended_action_values = np.array([base_action_values + [packet_value]])
        duration_predictor_output = self.duration_predictor.predict([extended_action_values, duration_values])
        mean = duration_predictor_output[0][0]
        stddev = math.exp(duration_predictor_output[0][1])
        duration = math.fabs(np.random.normal(mean, stddev))
        print("packet value")
        print(packet_value)
        print("duration value")
        print(duration)
        return [packet_value, duration]

    def learn(self):
        raw_packets = [item[0] for item in self.action_memory]
        raw_durations = [item[1] for item in self.action_memory]
        packet_size_1 = []
        packet_size_2 = []
        durations_1 = []
        for i in range(len(self.action_memory)):
            packets = [0] + raw_packets[0:i]
            durations = [0.0] + raw_durations[0:i]
            packet_size_1.append(packets)
            durations_1.append([[item] for item in durations])
        for i in range(len(self.action_memory)):
            packets = raw_packets[0:i+1]
            packet_size_2.append(packets)
        packet_outputs = np.zeros([len(raw_packets), self.n_actions])
        packet_outputs[np.arange(len(raw_packets)), raw_packets] = 1
        duration_outputs = np.array(raw_durations)
        mean = np.mean(self.reward_memory)
        std = np.std(self.reward_memory) if np.std(self.reward_memory) > 0 else 1
        advantages = (np.array(self.reward_memory) - mean) / std
        packets_size_input_1 = np.array(packet_size_1)
        packets_size_input_2 = np.array(packet_size_2)
        durations_input_1 = np.array(durations_1)
        for i in range(len(packets_size_input_1)):
            cost_1 = self.packet_size_policy.train_on_batch([np.array([packets_size_input_1[i]]), np.array([durations_input_1[i]]), np.array([advantages[i]])], np.array([packet_outputs[i]]))
            cost_2 = self.duration_policy.train_on_batch([np.array([packets_size_input_2[i]]), np.array([durations_input_1[i]]), np.array([advantages[i]])], np.array([duration_outputs[i]]))
        self.action_memory = []
        self.reward_memory = []
        return cost_1, cost_2

    def generate_step(self):
        action = self.choose_action(self.action_memory)
        self.action_memory.append(action)

    def generate_sequence(self):
        self.action_memory = []
        self.reward_memory = []
        while len(self.action_memory) < self.segment_length:
            self.generate_step()
        self.calculate_rewards()

    def get_sequence_generated(self):
        return self.action_memory

    def train(self):
        for epoch in range(self.epochs):
            self.generate_sequence()
            self.append_generated()
            self.learn()
            self.save_models()

    def append_generated(self):
        raw_packets = [item[0] for item in self.action_memory]
        raw_durations = [item[1] for item in self.action_memory]
        save_sequence(self.generated_packet_file, raw_packets)
        save_sequence(self.generated_durations_file, raw_durations)

    def save_models(self):
        self.packet_size_policy.save(self.psp_fn)
        self.packet_size_predictor.save(self.pspr_fn)
        self.duration_policy.save(self.dp_fn)
        self.duration_predictor.save(self.dpr_fn)

    def load_models(self):
        self.packet_size_policy = load_model(self.psp_fn, custom_objects={'SeqSelfAttention': SeqSelfAttention})
        self.packet_size_predictor = load_model(self.pspr_fn, custom_objects={'SeqSelfAttention': SeqSelfAttention})
        self.duration_policy = load_model(self.dp_fn, custom_objects={'SeqSelfAttention': SeqSelfAttention})
        self.duration_policy = load_model(self.dpr_fn, custom_objects={'SeqSelfAttention': SeqSelfAttention})
