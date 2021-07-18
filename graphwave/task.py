# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

class Task(object):
    def __init__(self, taskname):
        self.name = taskname

    def _classfication(self, embedding, labels_np, split_ratio=0.7):
        labels_np = shuffle(labels_np)
        nodes = labels_np[:, 0]
        labels = labels_np[:, 1]
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        train_size = int(labels_np.shape[0] * split_ratio)
        features = embedding[nodes]
        train_x = features[:train_size, :]
        train_y = labels[:train_size, :]
        test_x = features[train_size:, :]
        test_y = labels[train_size:, :]
        clf = OneVsRestClassifier(LogisticRegression(class_weight='balanced', solver='liblinear',max_iter=3000, n_jobs=1))
        #clf = OneVsRestClassifier(LogisticRegression(class_weight='balanced', solver='lbfgs', n_jobs=-1))
        # clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=100)
        clf.fit(train_x, train_y)
        y_pred = clf.predict_proba(test_x)
        y_pred = lb.transform(np.argmax(y_pred, 1))
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(test_y, 1))/len(y_pred)
        # fpr, tpr, thresholds = metrics.roc_curve(test_y, y_score)
        eval_dict = {'acc': acc,
                     'f1-micro': metrics.f1_score(np.argmax(test_y, 1), np.argmax(y_pred, 1), average='micro'),
                     'f1-macro': metrics.f1_score(np.argmax(test_y, 1), np.argmax(y_pred, 1), average='macro')}
        print(eval_dict)
        return eval_dict

    def classfication(self, embedding, labels_np, split_ratio=0.7, loop=1):
        eval_dict = {'acc': 0.0, 'f1-micro': 0.0, 'f1-macro': 0.0}
        for _ in range(loop):
            tmp_dict = self._classfication(embedding, labels_np, split_ratio)
            for key in tmp_dict.keys():
                eval_dict[key] += tmp_dict[key]
        for key in tmp_dict.keys():
            eval_dict[key] = round((1.0 * eval_dict[key]) / loop, 4)
        print("average performance:")
        print(eval_dict)
        return eval_dict

    def _link_prediction(self, embed, edgeList, labels, split_ratio=0.7, method='Hadamard'):
        #lb = LabelBinarizer()
        #labels = lb.fit_transform(labels)
        #print(embed.shape)
        ft = np.zeros((len(edgeList),embed.shape[1]))
        for i in range(len(edgeList)):
            src = edgeList[i][0]
            tgt = edgeList[i][1]
            if method == 'Hadamard':
                ft[i] = embed[src,:] * embed[tgt,:]
            elif method == 'Average':
                ft[i] == np.add(embed[src,:],embed[tgt,:])*0.5
        train_size = int(len(edgeList) * split_ratio)
        labels.reshape((-1))
        x_train = ft[:train_size, :]
        y_train = labels[:train_size]
        x_test = ft[train_size:, :]
        y_test = labels[train_size:]
        clf = LogisticRegression(class_weight='balanced', solver='liblinear',max_iter=5000)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_score = clf.predict_proba(x_test)[:, -1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        eval_dict = {'auc': metrics.auc(fpr, tpr),
                    'pr': metrics.average_precision_score(y_test, y_score),
                    'f1': metrics.f1_score(y_test, y_pred),
                    'f1-micro': metrics.f1_score(y_test, y_pred, average='micro'),
                    'f1-macro': metrics.f1_score(y_test, y_pred, average='macro')}
        print(eval_dict)
        return eval_dict

    def link_prediction(self, embedding, edgeList, labels, split_ratio=0.7, method='Hadamard', loop=100):
        eval_dict = {'auc': 0.0,'pr':0.0,'f1':0.0, 'f1-micro': 0.0, 'f1-macro': 0.0}
        for _ in range(loop):
            tmp_dict = self._link_prediction(embedding, edgeList, labels, split_ratio=0.7, method=method)
            for key in tmp_dict.keys():
                eval_dict[key] += tmp_dict[key]
        for key in tmp_dict.keys():
            eval_dict[key] = round((1.0 * eval_dict[key]) / loop, 4)
        print('average performance')
        print (eval_dict)
        return eval_dict
