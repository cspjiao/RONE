# coding=utf-8
import argparse
import os
import numpy as np


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='brazil-flights')
    args.add_argument('--method', type=str, default='struc2gauss')
    args.add_argument('--dimension', type=int, default=128)
    args.add_argument('--loop', type=int, default=100)
    args.add_argument('--workers', type=int, default=-1)
    args.add_argument('--no-classification', default=True, help='do classification', action='store_true')
    args.add_argument('--aggregator', type=str, default='simple')
    args.add_argument('--pruning-cutoff', type=float, default=0.5)
    args.add_argument('--recursive-iterations', type=int, default=3)
    args.add_argument('--bins', type=int, default=4)

    args.add_argument('--clusters', type=int, default=40)

    args.add_argument('--radius', type=int, default=2,
                      help='Maximum radius of ego-networks.')
    args.add_argument('--kernel', default='shortest_path',
                      help='Graph kernel (shortest_path or weisfeiler_lehman).')

    args.add_argument('-b', '--batch-size', type=int, default=16,
                      help='Number of training examples processed per step')
    return args.parse_args()


def main(args):
    method = args.method
    dimension = args.dimension
    command = ''
    #datasets = ['br-wiki-talk','cy-wiki-talk','eo-wiki-talk','gl-wiki-talk','ht-wiki-talk','oc-wiki-talk']
    datasets = ['brazil-flights','europe-flights','usa-flights','actor','reality-call','film']
    methods = ['ReFeX','RolX','RIDeRs-S','GraphWave','SEGK','struc2vec','struc2gauss','role2vec','node2bits','DRNE','GraLSP','GAS','RESD']
    for method in methods:
        for args.dataset in datasets:
            if method == 'role2vec':
                os.chdir('role2vec')
                command = 'python3 src/main.py --dataset clf/{} --dimensions {} --features motif' \
                          ' --motif-compression factorization --clusters {} --sampling second' \
                          ' --P 1 --Q 4 --alpha 0.5 --beta 0.5'.format(args.dataset, dimension, args.clusters)
            elif method == 'GraphWave':
                os.chdir('graphwave')
                command = 'python3 main.py --dataset clf/{}'.format(args.dataset)
            elif method == 'struc2vec':
                os.chdir('struc2vec')
                command = 'python3 src/main.py --OPT1 True --OPT2 True --OPT3 True --until-layer 6' \
                          ' --dimensions {} --dataset clf/{}'.format(dimension, args.dataset)
            elif method == 'node2bits':
                os.chdir('node2bits')
                command = 'python3 src/main.py --scope 4 --bin True --dataset clf/{} --dim {}'.format(args.dataset, dimension)
            elif method == 'ReFeX':
                os.chdir('RolX')
                if args.dataset in ['barbell']:
                    args.pruning_cutoff = 1
                command = 'python3 src/main.py --dataset clf/{} --aggregator {} --pruning-cutoff {} ' \
                          '--recursive-iterations {} --bins {}'.format(args.dataset, args.aggregator, args.pruning_cutoff,
                                                                       args.recursive_iterations, args.bins)
            elif method == 'RolX':
                os.chdir('RolX')
                command = 'python3 src/main.py --dataset clf/{} --dimensions {} --batch-size {} --epoch 170 --skip 1'.format(
                    args.dataset,
                    dimension, args.batch_size)
            elif method == 'RIDeRs-S':
                os.chdir('GLRD_and_RIDεRs')
                command = 'python3 -W ignore src/riders_right_sparse.py --dataset clf/{} --mdlit 8 --mrole 10'.format(
                    args.dataset)
            elif method == 'DRNE':
                os.chdir('DRNE')
                command = 'python3 src/main.py --dataset {} -s {} -b {}'.format(args.dataset, dimension, args.batch_size)
            elif method == 'SEGK':
                os.chdir('segk')
                if args.dataset == 'barbell':
                    dimension = 16
                command = 'python3 segk.py --dataset {} --dim {} --radius {} ' \
                          '--kernel {}'.format(args.dataset, dimension, args.radius, args.kernel)
            elif method == 'struc2gauss':
                os.chdir('struc2gauss')
                command = 'python2 main.py --dataset {} --dimension {}'.format(args.dataset, dimension)
            elif method =="GAS":
                os.chdir('GAS')
                command = 'python3 run.py --workers 4 --dataset clf/{} --n_h2 {}'.format(args.dataset,dimension)
            elif method =="RESD":
                os.chdir('RESD')
                command = 'python3 main.py --dataset {} --epoch 50'.format(args.dataset)
            elif method =="GraLSP":
                os.chdir('GraLSP')
                command = 'python3 main.py --dataset {} --embedding_dims {}'.format(args.dataset,dimension)
            print(command)
            os.system(command)
            os.chdir('../')

if __name__ == '__main__':
    args = parse_args()
    main(args)
