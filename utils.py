import numpy as np
from keras.utils.np_utils import to_categorical
from pcaputilities import extractSequences


def pad_seq(seq, max_length, PAD=0):
    """
    :param seq: list of int,
    :param max_length: int,
    :return seq: list of int,
    """
    seq += [PAD for i in range(max_length - len(seq))]
    return seq


def extract_all(real_packet_sizes_file):
    """
    Extract packet sequences from file of signed ints.
    Sign indicates direction
    # Arguments:
        real_packet_sizes_file: String
            path to file
    # Returns:
        normalized_packets: 2D list of unsigned ints
        V: vocab size
    """
    real_packets = extractSequences(real_packet_sizes_file)
    normalized_packets = []
    max_packet_size = 0
    for packets in real_packets:
        print(packets)
        max_packet_size = max(max([abs(int(x)) for x in packets]), max_packet_size)
    V = (max_packet_size * 2) + 3 # Add tokens PAD, BOS, EOS
    for packets in real_packets:
        packet_sizes = [(int(x) + max_packet_size + 3) for x in packets]
        normalized_packets.append(packet_sizes)
    return normalized_packets, V+1


def split_sequences(sequences, max_length):
    split_sequences = []
    for sequence in sequences:
        split_sequences = split_sequences + [sequence[i:i + max_length] for i in range(0, len(sequence), max_length)]
    return split_sequences


def build_generator_pretraining_datasets(sequences, V, PAD=0, BOS=1, EOS=2):
    """
    Format generator pretraining data batch.
    # Arguments:
        sequences: (B, None)
            real sequences with variable length
    # Returns:
        None: no input is needed for generator pretraining.
        x: numpy.array, shape = (B, max_length)
        y_true: numpy.array, shape = (B, max_length, V)
            labels with one-hot encoding.
            max_length is the max length of sequence in the batch.
            if length smaller than max_length, the data will be padded.
    """
    max_length = 0
    x, y_true = [], []
    for sequence in sequences:
        ids_x, ids_y_true = [], []
        ids_x.append(BOS)
        ids_x.extend(sequence)
        ids_x.append(EOS)
        x.append(ids_x)
        ids_y_true.extend(sequence)
        ids_y_true.append(EOS)
        y_true.append(ids_y_true)
        max_length = max(max_length, len(ids_x))

    for i, ids in enumerate(x):
        x[i] = x[i][:max_length]
    for i, ids in enumerate(y_true):
        y_true[i] = y_true[i][:max_length]

    x = [pad_seq(token, max_length, PAD) for token in x]
    x = np.array(x, dtype=np.int32)

    y_true = [pad_seq(token, max_length, PAD) for token in y_true]
    y_true = np.array(y_true, dtype=np.int32)
    y_true = to_categorical(y_true, num_classes=V)
    return x, y_true, max_length


def build_discriminator_datasets(real_sequences, fake_sequences, PAD=0, BOS=1, EOS=2):
    """
    Format discriminator data batch.
    # Arguments:
        real_sequences: (B, None)
            real sequences with variable length
        fake_seqences: (B, None)
            generated sequences with variable length
    # Returns:
        None: no input is needed for generator pretraining.
        x: numpy.array, shape = (B, max_length)
        y_true: numpy.array, shape = (B, max_length, V)
            labels with one-hot encoding.
            max_length is the max length of sequence in the batch.
            if length smaller than max_length, the data will be padded.
    """
    max_length = 0
    X, Y = [], []
    for real_sequence in real_sequences:
        x = [BOS]
        x.extend(real_sequence)
        x.append(EOS)
        X.append(x) # ex. [8, 10, 6, 3, EOS]
        Y.append(1)
        max_length = max(max_length, len(x))
    for fake_sequence in fake_sequences:
        x = [BOS]
        x.extend(fake_sequence)
        x.append(EOS)
        X.append(x)  # ex. [8, 10, 6, 3, EOS]
        Y.append(0)

    for i, ids in enumerate(X):
        X[i] = X[i][:max_length]

    X = [pad_seq(sen, max_length, PAD) for sen in X]
    X = np.array(X, dtype=np.int32)

    return X, np.array(Y, dtype=np.int32), max_length
