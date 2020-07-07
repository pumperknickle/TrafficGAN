from pcaputilities import signatureExtractionAll, featureExtractionAll, expandExtractAll
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
from keras.layers.merge import concatenate
from keras_self_attention import SeqSelfAttention
from keras.models import Model, load_model
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier

max_packet_size = 5000
total_actions = (max_packet_size * 2) + 1


def total_sigs(all_sigs):
    total = 0
    for key, value in all_sigs.items():
        total += len(value)
    return total


def extract(seq, var=0):
    all_packets = []
    for sample in seq:
        packet_size_sample = []
        for packet in sample:
            packet_size = packet[var]
            packet_size_sample.append(packet_size)
        all_packets.append(packet_size_sample)
    return all_packets


def extend_duration(all_durations):
    altered_durations = []
    for durations in all_durations:
        altered_duration = []
        for duration in durations:
            altered_duration.append([duration])
        altered_durations.append(altered_duration)
    return altered_durations


def randomlySample(sequences, num_seq, sample_size):
    sampled = []
    for _ in range(num_seq):
        chosen_sequence_index = np.random.randint(0, high=len(sequences))
        chosen_sequence = sequences[chosen_sequence_index]
        sequence_range_start = np.random.randint(0, high=len(chosen_sequence) - sample_size)
        sampled.append(chosen_sequence[sequence_range_start:sequence_range_start+sample_size])
    return sampled


class Discriminator(object):

    def __init__(self, generated=[], real=[], segment_length=200, batch_size=32, epochs=20, alpha=0.0005, distance_threshold=5.0,
                 cluster_threshold=4, min_sig_size=1, max_sig_size=5, signature_extraction_model=None,
                 lstm_discriminator_network=None, n_actions=total_actions, embedding_dim=100, lstm_units=100,
                 dense_units=100, sig_fn='sigfn.h5', lstm_fn='lstm.h5'):
        self.segment_length = segment_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = alpha
        self.generated = generated
        self.real = real
        self.randomly_sampled_reals = randomlySample(self.real, len(generated), segment_length)
        self.distance_threshold = distance_threshold
        self.cluster_threshold = cluster_threshold
        self.min_sig_size = min_sig_size
        self.max_sig_size = max_sig_size
        self.real_packet_sizes = extract(self.randomly_sampled_reals)
        self.generated_packet_sizes = extract(generated)
        self.real_durations = extract(self.randomly_sampled_reals, 1)
        self.generated_durations = extract(generated, 1)
        self.all_signatures = signatureExtractionAll(extract(real), min_sig_size, max_sig_size, distance_threshold,
                                                     cluster_threshold)
        self.total_sigs = total_sigs(self.all_signatures)
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.sig_fn = sig_fn
        self.lstm_fn = lstm_fn
        if signature_extraction_model is None or lstm_discriminator_network is None:
            self.signature_extraction_network = self.signature_extraction_network()
            self.lstm_discriminator_network = self.lstm_discriminator_network()
        else:
            self.signature_extraction_network = signature_extraction_model
            self.lstm_discriminator_network = lstm_discriminator_network

    # Define the network model that discriminates
    def signature_extraction_network(self):
        model = Sequential()
        model.add(Dense(self.dense_units, activation='relu', input_shape=(self.total_sigs,)))
        model.add(Dense(self.dense_units, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def lstm_discriminator_network(self):
        packets_input = Input(shape=(self.segment_length,))
        packet_embedding = Embedding(input_dim=self.n_actions, output_dim=self.embedding_dim)(packets_input)
        durations_input = Input(shape=(self.segment_length, 1))
        merged = concatenate([packet_embedding, durations_input])
        lstm1 = LSTM(self.lstm_units, return_sequences=True)(merged)
        attention = SeqSelfAttention(attention_activation='sigmoid')(lstm1)
        lstm2 = LSTM(self.lstm_units)(attention)
        hidden_dense = Dense(self.dense_units, activation='relu')(lstm2)
        packets_output = Dense(2, activation='softmax')(hidden_dense)
        model = Model(inputs=[packets_input, durations_input], outputs=packets_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        real_features = featureExtractionAll(self.real_packet_sizes, self.all_signatures)
        generated_features = featureExtractionAll(self.generated_packet_sizes, self.all_signatures)
        labels = [1] * len(real_features) + [0] * len(generated_features)
        categorical_labels = np_utils.to_categorical(labels)
        all_durations = self.real_durations + self.generated_durations
        for _ in range(self.epochs):
            self.lstm_discriminator_network.train_on_batch([np.array(self.real_packet_sizes + self.generated_packet_sizes), np.array(extend_duration(all_durations))], categorical_labels)
            self.signature_extraction_network.train_on_batch(np.array(real_features + generated_features), categorical_labels)
        self.save_models()

    def predict(self, seqs):
        packet_sizes = extract(seqs)
        features = featureExtractionAll(packet_sizes, self.all_signatures)
        sen_output = self.signature_extraction_network.predict(np.array(features))
        durations = extend_duration(extract(seqs, var=1))
        lstm_output = self.lstm_discriminator_network.predict([np.array(packet_sizes), np.array(durations)])
        predictions = []
        for i in range(len(packet_sizes)):
            predictions.append((sen_output[i] + lstm_output[i][1])/2.0)
            print("adv")
            print((sen_output[i][1] + lstm_output[i][1])/2.0)
        return predictions

    def save_models(self):
        self.signature_extraction_network.save(self.sig_fn)
        self.lstm_discriminator_network.save(self.lstm_fn)

    def load_models(self):
        self.signature_extraction_network = load_model(self.sig_fn)
        self.lstm_discriminator_network = load_model(self.lstm_fn, custom_objects={'SeqSelfAttention': SeqSelfAttention})
