# GAN loop
import csv
import math

from Generator import Generator
from Discriminator import Discriminator
import numpy as np
import random
from pcaputilities import extractSequences
from keras.utils import plot_model
import sys
import glob
import tensorflow as tf

from pcaputilities import convert_to_ps_durations, convertToFeatures

# directory = sys.argv[1]
# pcapPath = directory + '/*.pcap'
# pcapFiles = glob.glob(pcapPath)
# all_durations = []
# all_ps_sequences = []
# for file in pcapFiles:
#     durations = convert_to_ps_durations(file)
#     sequences = convertToFeatures(file)
#     all_durations.append(durations)
#     all_ps_sequences.append(sequences)

# def save_sequences(filename, sequences):
#     with open(filename, 'w', newline='\n') as csvfile:
#         csv_writer = csv.writer(csvfile, delimiter=' ')
#         for sequence in sequences:
#             csv_writer.writerow(sequence)


real_packet_sizes_file = 'real_packet_sizes.txt'
real_durations_file = 'real_durations.txt'

# # save_sequences(real_packet_sizes_file, all_ps_sequences)
# # save_sequences(real_durations_file, all_durations)

real_packets = extractSequences(real_packet_sizes_file)
real_durations = extractSequences(real_durations_file)

max_packet_size = 5000
total_actions = (max_packet_size * 2) + 1
segment_length=200

gen_features = []
real_features = []

for real_packet in real_packets:
    segment_length = min(len(real_packet), segment_length)

for i in range(len(real_packets)):
    durations = [float(x) for x in real_durations[i]]
    packet_sizes = [(int(x) + max_packet_size + 1) for x in real_packets[i]]
    mean_d = np.mean(durations)
    std_d = np.std(durations)
    gen_sample = []
    real_sample = []
    for j in range(segment_length):
        packet_size_rand = random.randint(1, total_actions - 1)
        duration_rand = math.fabs(np.random.normal(mean_d, std_d))
        gen_sample.append([packet_size_rand, duration_rand])
    for j in range(len(real_packets[i])):
        real_sample.append([packet_sizes[j], durations[j]])
    gen_features.append(gen_sample)
    real_features.append(real_sample)

# Pretraining
discriminator = Discriminator(gen_features, real_features, segment_length)
# discriminator.load_models()
discriminator.train()

generator = Generator(discriminator, segment_length)
generator.train()

plot_model(discriminator.lstm_discriminator_network, to_file='lstm_discriminator.png', show_shapes=True, show_layer_names=True)
plot_model(discriminator.signature_extraction_network, to_file='signature_extraction_discriminator.png', show_shapes=True, show_layer_names=True)
plot_model(generator.build_duration_policy_network()[0], to_file='duration_policy.png', show_shapes=True, show_layer_names=True)
plot_model(generator.build_duration_policy_network()[1], to_file='duration_predictor.png', show_shapes=True, show_layer_names=True)
plot_model(generator.build_packet_size_policy_network()[0], to_file='packet_size_policy.png', show_shapes=True, show_layer_names=True)
plot_model(generator.build_packet_size_policy_network()[1], to_file='packet_size_predictor.png', show_shapes=True, show_layer_names=True)

# Adverserial Loop
for i in range(100):
    print("outer loop")
    print("i")
    generator.train()
    fake_sequences = []
    for _ in range(len(real_features)):
        generator.generate_sequence()
        sequence_gen = generator.get_sequence_generated()
        fake_sequences.append(sequence_gen)
    discriminator = Discriminator(fake_sequences, real_features, segment_length)
    discriminator.train()
    generator.discriminator = discriminator




