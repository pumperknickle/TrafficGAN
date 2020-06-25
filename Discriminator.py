from pcaputilities import signatureExtractionAll, featureExtractionAll, expandExtractAll
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding, Bidirectional, Lambda
from keras.layers.merge import concatenate
from keras_self_attention import SeqSelfAttention
from keras.models import Model, load_model
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from pcaputilities import extractSequences

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

# Expands a sequence - [1,2,3,4] into [1], [1,2], [1,2,3], [1,2,3,4]
def expand(seq):
    all_packets = []
    for sample in seq:
        for i in range(len(sample)):
            packets = sample[0:i + 1]
            all_packets.append(packets)
    return all_packets


def order_by_seq_len(packet_sizes, durations, is_reals):
    all_ps = dict()
    all_d = dict()
    all_reals = dict()
    final_ps = []
    final_d = []
    final_real = []
    for i in range(len(packet_sizes)):
        packets = packet_sizes[i]
        packet_len = len(packets)
        if packet_len not in all_ps:
            all_ps[packet_len] = []
        if packet_len not in all_d:
            all_d[packet_len] = []
        if packet_len not in all_reals:
            all_reals[packet_len] = []
        duration = durations[i]
        if len(duration) != packet_len:
            print("error")
            return
        is_real = is_reals[i]
        all_ps[packet_len].append(packets)
        all_d[packet_len].append(duration)
        all_reals[packet_len].append(is_real)
    for key, value in all_ps.items():
        final_ps.append(value)
        final_d.append(all_d[key])
        final_real.append(all_reals[key])
    return final_ps, final_d, final_real


class Discriminator(object):

    def __init__(self, generated=[], real=[], batch_size=32, epochs=1, alpha=0.001, distance_threshold=5.0,
                 cluster_threshold=4, min_sig_size=1, max_sig_size=5, signature_extraction_model=None,
                 lstm_discriminator_network=None, n_actions=total_actions, embedding_dim=100, lstm_units=100,
                 dense_units=100, sig_fn='sigfn.h5', lstm_fn='lstm.h5'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = alpha
        self.generated = generated
        self.real = real
        self.distance_threshold = distance_threshold
        self.cluster_threshold = cluster_threshold
        self.min_sig_size = min_sig_size
        self.max_sig_size = max_sig_size
        self.real_packet_sizes = extract(real)
        self.generated_packet_sizes = extract(generated)
        self.expanded_real = expand(real)
        self.expanded_generated = expand(generated)
        self.expanded_real_packet_sizes = extract(self.expanded_real)
        self.expanded_generated_packet_sizes = extract(self.expanded_generated)
        self.expanded_real_durations = extract(self.expanded_real, 1)
        self.expanded_generated_durations = extract(self.expanded_generated, 1)
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

    # Define the network model that generates packet sizes and direction
    def signature_extraction_network(self):
        model = Sequential()
        model.add(Dense(self.dense_units, activation='relu', input_shape=(self.total_sigs,)))
        model.add(Dense(self.dense_units, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def lstm_discriminator_network(self):
        packets_input = Input(shape=(None,))
        packet_embedding = Embedding(input_dim=self.n_actions, output_dim=self.embedding_dim)(packets_input)
        durations_input = Input(shape=(None, 1))
        merged = concatenate([packet_embedding, durations_input])
        lstm1 = LSTM(self.lstm_units, return_sequences=True)(merged)
        attention = SeqSelfAttention(attention_activation='sigmoid')(lstm1)
        lstm2 = LSTM(self.lstm_units)(attention)
        hidden_dense = Dense(self.dense_units, activation='relu')(lstm2)
        packets_output = Dense(2, activation='softmax')(hidden_dense)
        model = Model(inputs=[packets_input, durations_input], outputs=packets_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def score(self):
        real_features = expandExtractAll(self.real_packet_sizes, self.all_signatures)
        generated_features = expandExtractAll(self.generated_packet_sizes, self.all_signatures)
        labels = [0] * len(real_features) + [1] * len(generated_features)
        categorical_labels = np_utils.to_categorical(labels)
        signature_estimator = KerasClassifier(build_fn=self.signature_extraction_network, epochs=200, batch_size=5,
                                              verbose=1)
        kfold = KFold(n_splits=10, shuffle=True)
        signature_results = cross_val_score(signature_estimator, np.array(real_features + generated_features),
                                            categorical_labels, cv=kfold)
        print("Signature Network Results")
        print("Baseline: %.2f%% (%.2f%%)" % (signature_results.mean() * 100, signature_results.std() * 100))
        lstm_estimator = KerasClassifier(build_fn=self.lstm_discriminator_network, epochs=200, batch_size=5, verbose=1)
        lstm_results = cross_val_score(lstm_estimator, [
            np.array(self.expanded_real_packet_sizes + self.expanded_generated_packet_sizes),
            np.array(self.expanded_real_durations + self.expanded_generated_durations)], categorical_labels, cv=kfold)
        print("LSTM Network Results")
        print("Baseline: %.2f%% (%.2f%%)" % (lstm_results.mean() * 100, lstm_results.std() * 100))

    def train(self):
        all_durations = self.expanded_real_durations + self.expanded_generated_durations
        altered_durations = extend_duration(all_durations)
        # altered_durations = []
        # for durations in all_durations:
        #     altered_duration = []
        #     for duration in durations:
        #         altered_duration.append([duration])
        #     altered_durations.append(altered_duration)
        real_features = expandExtractAll(self.real_packet_sizes, self.all_signatures)
        generated_features = expandExtractAll(self.generated_packet_sizes, self.all_signatures)
        labels = [0] * len(real_features) + [1] * len(generated_features)
        categorical_labels = np_utils.to_categorical(labels)
        ordered_ps, ordered_d, ordered_r = order_by_seq_len(
            self.expanded_real_packet_sizes + self.expanded_generated_packet_sizes, altered_durations,
            categorical_labels.tolist())
        for i in range(len(ordered_ps)):
            self.lstm_discriminator_network.fit([np.array(ordered_ps[i]), np.array(ordered_d[i])], np.array(ordered_r[i]),
                                                epochs=self.epochs, verbose=2)
            self.save_models()
        self.signature_extraction_network.fit(np.array(real_features + generated_features), categorical_labels,
                                              epochs=self.epochs)
        self.save_models()

    def predict(self, seqs):
        packet_sizes = extract(seqs)
        features = featureExtractionAll(packet_sizes, self.all_signatures)
        sen_output = self.signature_extraction_network.predict(np.array(features))
        durations = extend_duration(extract(seqs, var=1))
        lstm_output = []
        for i in range(len(packet_sizes)):
            lstm_output.append(self.lstm_discriminator_network.predict([np.array([packet_sizes[i]]), np.array([durations[i]])]))
        predictions = []
        for i in range(len(packet_sizes)):
            predictions.append((sen_output[i][0] + lstm_output[i][0][0])/2.0)
        return predictions

    def save_models(self):
        self.signature_extraction_network.save(self.sig_fn)
        self.lstm_discriminator_network.save(self.lstm_fn)

    def load_models(self):
        self.signature_extraction_network = load_model(self.sig_fn)
        self.lstm_discriminator_network = load_model(self.lstm_fn, custom_objects={'SeqSelfAttention': SeqSelfAttention})
        # real_packet_sizes_file = 'real_packet_sizes.txt'
        # real_durations_file = 'real_durations.txt'
        # real_packets = extractSequences(real_packet_sizes_file)
        # real_durations = extractSequences(real_durations_file)
        # real_features = []
        # for i in range(len(real_packets)):
        #     durations = [float(x) for x in real_durations[i]]
        #     packet_sizes = [(int(x) + max_packet_size + 1) for x in real_packets[i]]
        #     real_sample = []
        #     for j in range(len(real_packets[i])):
        #         real_sample.append([packet_sizes[j], durations[j]])
        #     real_features.append(real_sample)
        # self.all_signatures = signatureExtractionAll(extract(real_features), self.min_sig_size, self.max_sig_size,
        #                                              self.distance_threshold, self.cluster_threshold)
