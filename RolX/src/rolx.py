import math
import time
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import tensorflow as tf
from layers import Factorization
from refex import RecursiveExtractor
from print_and_read import log_setup, tab_printer, epoch_printer, log_updater, data_reader, \
    data_saver
import pandas as pd

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class Model(object):
    """
    Abstract model class.
    """

    def __init__(self, args):
        """
        Every model needs the same initialization -- args, graph.
        We delete the sampler object to save memory.
        We also build the computation graph up. 
        """
        self.args = args
        if self.args.skip == 0:
            print("generate ReFeX...")
            self.recurser = RecursiveExtractor(args)
            self.dataset = np.array(self.recurser.new_features)
        else:
            print("reading ../embed/ReFeX")
            try:
                self.dataset = pd.read_csv("../embed/ReFeX/" + args.dataset + '_128.emb',encoding='utf-8', sep=',')
            except Exception as e:
                self.dataset  = pd.read_csv("../embed/ReFeX/" + args.dataset + ".emb",encoding='utf-8', sep=',')

            # self.dataset=np.load('../Role2vec/data/RolX/output/test.npy')
            self.dataset = self.dataset.drop(['id'], axis=1).values
        self.user_size = self.dataset.shape[0]
        self.feature_size = self.dataset.shape[1]
        self.nodes = list(range(0, self.user_size))
        self.true_step_size = (self.user_size * self.args.epochs) / self.args.batch_size
        self.build()

    def build(self):
        """
        Building the model.
        """
        pass

    def feed_dict_generator(self):
        """
        Creating the feed generator
        """
        pass

    def train(self):
        """
        Training the model.
        """
        pass


class ROLX(Model):
    """
    ROLX class.
    """

    def build(self):
        """
        Method to create the computational graph.
        """
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            self.factorization_layer = Factorization(self.args, self.user_size,
                                                     self.feature_size)
            self.loss = self.factorization_layer()
            self.batch = tf.Variable(0)
            self.step = tf.compat.v1.placeholder("float")
            self.learning_rate_new = tf.compat.v1.train.polynomial_decay(
                self.args.initial_learning_rate,
                self.batch,
                self.true_step_size,
                self.args.minimal_learning_rate,
                self.args.annealing_factor)
            self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate_new).minimize(
                self.loss,
                global_step=self.batch)
            self.init = tf.compat.v1.global_variables_initializer()
            #self.train()

    def feed_dict_generator(self, nodes, step):
        """
        Method to generate left and right handside matrices, proper time index and overlap
        vector.
        """
        left_nodes = np.array(nodes)
        right_nodes = np.array([i for i in range(0, self.feature_size)])

        targets = self.dataset[nodes, :]

        feed_dict = {self.factorization_layer.edge_indices_left: left_nodes,
                     self.factorization_layer.edge_indices_right: right_nodes,
                     self.factorization_layer.target: targets,
                     self.step: float(step)}

        return feed_dict

    def train(self):
        """
        Method for training the embedding, logging.
        """
        self.current_step = 0
        self.log = log_setup(self.args)

        with tf.compat.v1.Session(graph=self.computation_graph, config=config) as session:
            self.init.run()
            print("Model Initialized.")

            for repetition in range(0, self.args.epochs):
                random.shuffle(self.nodes)
                self.optimization_time = 0
                self.average_loss = 0

                epoch_printer(repetition)
                for i in tqdm(range(0, int(len(self.nodes) / self.args.batch_size))):
                    self.current_step = self.current_step + 1
                    feed_dict = self.feed_dict_generator(
                        self.nodes[i * self.args.batch_size:(i + 1) * self.args.batch_size],
                        self.current_step)
                    start = time.time()
                    _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)
                    end = time.time()
                    self.optimization_time = self.optimization_time + (end - start)
                    self.average_loss = self.average_loss + loss

                print("")
                self.average_loss = (self.average_loss / i)
                self.log = log_updater(self.log, repetition, self.average_loss,
                                       self.optimization_time)
                # tab_printer(self.log)
            self.features = self.factorization_layer.embedding_node.eval()
            print(self.features.shape)
            embedding = self.features
            return embedding
            # data_saver(self.dataset[:,1:], "../cache/features/{}_features.csv".format(
            #     self.args.dataset.replace('clf/', '')))
