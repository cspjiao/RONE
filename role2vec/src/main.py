from parser import parameter_parser
from role2vec import Role2Vec
from utils import tab_printer
from task import Task
import numpy as np
import pandas as pd
import pickle as pkl
import random
import time

def main(args):
    """
    Role2Vec model fitting.
    :param args: Arguments object.
    """
    tab_printer(args)
    model = Role2Vec(args)
    t1 = time.time()
    print("Start time: {}".format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))))
    model.do_walks()
    model.create_structural_features()
    model.learn_embedding()
    t2 = time.time()
    print("Embedding time: {}".format(t2 - t1))
    timepath = '../runningtime.txt'
    f = open(timepath, "a+")
    f.write("Role2Vec {} running time :{}\n".format(args.dataset, t2 - t1))
    f.close()
    model.save_embedding()


if __name__ == "__main__":
    args = parameter_parser()
    main(args)
