import os

import numpy as np
import pandas as pd
import tensorflow as tf
from models import *
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ''  # 指定第一块GPU可用
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # 程序按需申请内存
flags = tf.app.flags

flags.DEFINE_string("dataset", "brazil-flights", "The dataset to use, corresponds to a folder under data/")
flags.DEFINE_integer("path_length", 10, "The length of random_walks")
flags.DEFINE_integer("num_paths", 100, "The number of paths to use per node")
flags.DEFINE_integer("window_size", 5, "The window size to sample neighborhood")
flags.DEFINE_integer("batch_size", 10, "batch size")
flags.DEFINE_integer("neg_size", 8, "negative sampling size")
flags.DEFINE_float("learning_rate", 0.005, "learning rate")
flags.DEFINE_string("optimizer", "Adam", "The optimizer to use")
flags.DEFINE_integer("embedding_dims", 128, "The size of each embedding")
flags.DEFINE_integer("num_steps", 5001, "Steps to train")
flags.DEFINE_integer("num_skips", 5, "how many samples to draw from a single walk")
flags.DEFINE_integer("num_neighbor", 10, "How many neighbors to sample, for graphsage")
flags.DEFINE_integer("hidden_dim", 100, "The size of hidden dimension, for graphsage")
flags.DEFINE_integer("walk_dim", 30, "The size of embeddings for anonym. walks.")
flags.DEFINE_integer("anonym_walk_len", 8, "The length of each anonymous walk, 4 or 5")
flags.DEFINE_float("walk_loss_lambda", 0.1, "Weight of loss focusing on anonym walk similarity")
flags.DEFINE_string("purpose", "none", "Tasks for evaluation, classification or link_prediction")
flags.DEFINE_float("linkpred_ratio", 0.1, "The ratio of edges being removed for link prediction")
flags.DEFINE_float("p", 0.25, "return parameter for node2vec walk")
flags.DEFINE_float("q", 1, "out parameter for node2vec walk")
flags.DEFINE_integer("inductive", 0, "whether to do inductive inference")
flags.DEFINE_integer("inductive_model_epoch", None, "the epoch of the saved model")
flags.DEFINE_string("inductive_model_name", None, "the path towards the loaded model")

FLAGS = flags.FLAGS


def main(_):
    print("Dataset: %s" % FLAGS.dataset)
    print("hidden dimension: %d" % FLAGS.hidden_dim)
    print("Lambda for walk loss: %f" % FLAGS.walk_loss_lambda)
    print("Anonym walk length: %d" % FLAGS.anonym_walk_len)
    sess = tf.Session()

    save_path = "../embed/GraLSP/clf"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    t1 = time.time()
    print("Start time: {}".format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))))
    model = GraLSP(sess, FLAGS.dataset, FLAGS.purpose, FLAGS.linkpred_ratio, FLAGS.path_length,
                   FLAGS.num_paths, FLAGS.window_size, FLAGS.batch_size, FLAGS.neg_size, FLAGS.learning_rate,
                   FLAGS.optimizer, FLAGS.embedding_dims, save_path, FLAGS.num_steps, FLAGS.num_skips, FLAGS.hidden_dim,
                   FLAGS.num_neighbor, FLAGS.anonym_walk_len, FLAGS.walk_loss_lambda, FLAGS.walk_dim,
                   FLAGS.p, FLAGS.q)
    # if not FLAGS.inductive:
    #     model.train()
    # else:
    #     model.restore(FLAGS.inductive_model_name)
    model.train()
    embedding = model.get_full_embeddings()
    t2 = time.time()
    print("Embedding time: {}".format(t2 - t1))
    timepath = '../runningtime.txt'
    f = open(timepath, "a+")
    f.write("GraLSP {} running time :{}\n".format(FLAGS.dataset, t2 - t1))
    f.close()
    columns = ['id'] + ['x_' + str(i) for i in range(embedding.shape[1])]
    idx = np.arange(embedding.shape[0]).reshape(-1, 1)
    embedding = pd.DataFrame(np.concatenate([idx, embedding], axis=1), columns=columns)
    embedding = embedding.sort_values(by='id')
    print("Saving embedding to " + '{}/{}_{}.emb'.format(save_path, FLAGS.dataset,FLAGS.embedding_dims))
    embedding.to_csv('{}/{}_{}.emb'.format(save_path, FLAGS.dataset,FLAGS.embedding_dims), index=False)
    # model.save_embeddings(FLAGS.inductive_model_epoch, save_model=False)


if __name__ == "__main__":
    tf.app.run()
