from utils import extract_all, build_generator_pretraining_datasets, build_discriminator_datasets, split_sequences
from rl import Agent, Environment
from models import Discriminator, GeneratorPretraining
from keras.optimizers import Adam
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class Trainer(object):
    def __init__(self, B, T, g_E, g_H, d_E, d_H, d_dropout, g_lr=1e-3, d_lr=1e-3,
                 n_sample=16, generate_samples=20, init_eps=0.1, real_packets_file="real_packet_sizes.txt"):
        self.top = os.getcwd()
        self.B, self.T = B, T
        self.g_E, self.g_H = g_E, g_H
        self.d_E, self.d_H = d_E, d_H
        self.d_dropout = d_dropout
        self.generate_samples = generate_samples
        self.g_lr, self.d_lr = g_lr, d_lr
        self.eps = init_eps
        self.init_eps = init_eps
        self.real_packets = real_packets_file
        self.pos_sequences, self.V = extract_all(real_packets_file)
        self.pos_sequences = split_sequences(self.pos_sequences, T-2)
        self.neg_sequences = []
        self.agent = Agent(B, self.V, g_E, g_H, g_lr)
        self.g_beta = Agent(B, self.V, g_E, g_H, g_lr)
        self.discriminator = Discriminator(self.V, d_E, d_H, d_dropout)
        self.env = Environment(self.discriminator, self.B, self.V, self.g_beta, n_sample=n_sample)
        self.generator_pre = GeneratorPretraining(self.V, g_E, g_H)
        self.g_pre_path, self.d_pre_path = None, None

    def pre_train(self, g_epochs=3, d_epochs=1, g_pre_path=None, d_pre_path=None, g_lr=1e-3, d_lr=1e-3):
        self.pre_train_generator(g_epochs=g_epochs, g_pre_path=g_pre_path, lr=g_lr)
        self.pre_train_discriminator(d_epochs=d_epochs, d_pre_path=d_pre_path, lr=d_lr)

    def pre_train_generator(self, g_epochs=3, lr=1e-3, g_pre_path=None):
        if g_pre_path is None:
            self.g_pre_path = os.path.join(self.top, 'data', 'save', 'generator_pre.hdf5')
        else:
            self.g_pre_path = g_pre_path

        g_adam = Adam(lr)
        self.generator_pre.compile(g_adam, 'categorical_crossentropy')
        print('Generator pre-training')
        self.generator_pre.summary()
        X, Y, self.T = build_generator_pretraining_datasets(self.pos_sequences, self.V)
        self.generator_pre.fit(x=X, y=Y, batch_size=self.B, epochs=g_epochs)
        self.generator_pre.save_weights(self.g_pre_path)
        self.reflect_pre_train()

    def pre_train_discriminator(self, d_epochs=1, lr=1e-3, d_pre_path=None):
        if d_pre_path is None:
            self.d_pre_path = os.path.join(self.top, 'data', 'save', 'discriminator_pre.hdf5')
        else:
            self.d_pre_path = d_pre_path

        neg_sequences = self.agent.generator.generate_samples(self.T, self.generate_samples)
        X, Y, _ = build_discriminator_datasets(self.pos_sequences, neg_sequences)
        d_adam = Adam(lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()
        print('Discriminator pre-training')
        self.discriminator.fit(x=X, y=Y, batch_size=self.B, epochs=d_epochs)
        self.discriminator.save(self.d_pre_path)

    def reflect_pre_train(self):
        for layer in self.generator_pre.layers:
            print(len(layer.get_weights()))
        for layer in self.agent.generator.packet_size_policy.layers:
            print(len(layer.get_weights()))
        w1 = self.generator_pre.layers[1].get_weights()
        w2 = self.generator_pre.layers[2].get_weights()
        w3 = self.generator_pre.layers[3].get_weights()
        self.agent.generator.packet_size_policy.layers[1].set_weights(w1)
        self.g_beta.generator.packet_size_policy.layers[1].set_weights(w1)
        self.agent.generator.packet_size_policy.layers[4].set_weights(w2)
        self.g_beta.generator.packet_size_policy.layers[4].set_weights(w2)
        self.agent.generator.packet_size_policy.layers[5].set_weights(w3)
        self.g_beta.generator.packet_size_policy.layers[5].set_weights(w3)

    def load_pre_train(self, g_pre_path, d_pre_path):
        self.load_pre_train_g(g_pre_path)
        self.load_pre_train_d(d_pre_path)

    def load_pre_train_g(self, g_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()

    def load_pre_train_d(self, d_pre_path):
        self.discriminator.load_weights(d_pre_path)

    def save(self, g_path, d_path):
        self.agent.save(g_path)
        self.discriminator.save(d_path)

    def load(self, g_path, d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator.load_weights(d_path)

    def train(self, steps=10, g_steps=1, d_steps=1, d_epochs=3, g_weights_path='data/save/generator.pkl',
              d_weights_path='data/save/discriminator.hdf5', verbose=True, head=1):
        d_adam = Adam(self.d_lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy', metrics=['accuracy'])
        self.eps = self.init_eps
        for step in range(steps):
            print("Adverserial Training - Generator")
            for _ in range(g_steps):
                rewards = np.zeros([self.B, self.T])
                self.agent.reset()
                self.env.reset()
                for t in range(self.T):
                    state = self.env.get_state()
                    action = self.agent.act(state, epsilon=0.0, stateful=False)
                    next_state, reward, is_episode_end, info = self.env.step(action)
                    self.agent.generator.update(state, action, reward)
                    rewards[:, t] = reward.reshape([self.B, ])
            print("Adverserial Training - Discriminator")
            for _ in range(d_steps):
                neg_sequences = self.agent.generator.generate_samples(self.T, self.generate_samples)
                print("generated sequences")
                print(neg_sequences)
                X, Y, _ = build_discriminator_datasets(self.pos_sequences, neg_sequences)
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=25)
                self.discriminator.fit(x=X_train, y=y_train, batch_size=self.B, epochs=d_epochs, validation_data=(X_test, y_test))

            # Update env.g_beta to agent
            self.agent.save(g_weights_path)
            self.g_beta.load(g_weights_path)

            self.discriminator.save(d_weights_path)
            self.eps = max(self.eps * (1 - float(step) / steps * 4), 1e-4)


tf.config.experimental_run_functions_eagerly(True)

# hyper parameters
B = 32   # batch size
T = 50   # Max length of sentence
g_E = 64   # Generator embedding size
g_H = 64   # Generator LSTM hidden size
g_lr = 1e-5
d_E = 64   # Discriminator embedding and Highway network sizes
d_H = 64
d_dropout = 0.0   # Discriminator dropout ratio
d_lr = 1e-6

n_sample = 16   # Number of Monte Calro Search
generate_samples = 700   # Number of generated sentences

# Pretraining parameters
g_pre_lr = 1e-2
d_pre_lr = 1e-4
g_pre_epochs = 60
d_pre_epochs = 1

top = os.getcwd()
g_pre_weights_path = os.path.join(top, 'data', 'save', 'generator_pre.hdf5')
d_pre_weights_path = os.path.join(top, 'data', 'save', 'discriminator_pre.hdf5')
g_weights_path = os.path.join(top, 'data', 'save', 'generator.pkl')
d_weights_path = os.path.join(top, 'data', 'save', 'discriminator.hdf5')

trainer = Trainer(B, T, g_E, g_H, d_E, d_H, d_dropout, g_lr=g_lr, d_lr=d_lr, n_sample=n_sample, generate_samples=generate_samples)

# Pretraining for adversarial training
trainer.pre_train(g_epochs=g_pre_epochs, d_epochs=d_pre_epochs, g_pre_path=g_pre_weights_path, d_pre_path=d_pre_weights_path, g_lr=g_pre_lr, d_lr=d_pre_lr)

trainer.load_pre_train(g_pre_weights_path, d_pre_weights_path)
trainer.reflect_pre_train()

trainer.train(steps=100, g_steps=1, head=10)


