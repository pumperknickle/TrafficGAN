# GAN loop
import csv
from Generator import Generator
from Discriminator import Discriminator
import numpy as np
import random
from pcaputilities import extractSequences
import sys
import glob

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

gen_features = []
real_features = []

for i in range(len(real_packets)):
    durations = [float(x) for x in real_durations[i]]
    packet_sizes = [(int(x) + max_packet_size + 1) for x in real_packets[i]]
    mean_d = np.mean(durations)
    std_d = np.std(durations)
    gen_sample = []
    real_sample = []
    for j in range(len(real_packets[i])):
        packet_size_rand = random.randint(1, total_actions - 1)
        duration_rand = np.random.normal(mean_d, std_d)
        gen_sample.append([packet_size_rand, duration_rand if duration_rand > 0 else -1 * duration_rand])
        real_sample.append([packet_sizes[j], durations[j]])
    gen_features.append(gen_sample)
    real_features.append(real_sample)

# Pretraining
discriminator = Discriminator(gen_features, real_features)
# discriminator.load_models()
discriminator.train()

generator = Generator(discriminator)
generator.train()

# Adverserial Loop
for i in range(10):
    print("outer loop")
    print("i")
    generator.train()
    fake_sequences = []
    for _ in range(len(real_features)):
        generator.generate_sequence()
        sequence_gen = generator.get_sequence_generated()
        fake_sequences.append(sequence_gen)
    discriminator = Discriminator(fake_sequences, real_features)
    discriminator.train()
    generator.discriminator = discriminator




