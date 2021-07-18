import random
import numpy as np

from sklearn import svm
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, homogeneity_score, completeness_score, silhouette_score


def evaluate_clustering(embeddings, y):
    homogeneity = list()
    completeness = list()
    silhouette = list()
    unique = np.unique(y)
    for embedding_matrix in embeddings:
        cl = AgglomerativeClustering(n_clusters=unique.size, linkage='single')
        cl.fit(embedding_matrix)
        labels_pred = cl.labels_

        homogeneity.append(homogeneity_score(labels_pred, y))
        completeness.append(completeness_score(labels_pred, y))
        silhouette.append(silhouette_score(embedding_matrix, labels_pred))
    return homogeneity, completeness, silhouette


def evaluate_classification(embeddings, y):
    avg_accs = list()
    avg_f1 = list()
    
    for i in range(len(embeddings)):
        avg_accs.append(list())
        avg_f1.append(list())

    for i in range(100):
        kf = KFold(n_splits=10, shuffle=True)
        accs = list()
        f1 = list()
        
        for j in range(len(embeddings)):
            accs.append(list())
            f1.append(list())
        
        for train_index, test_index in kf.split(embeddings[0]):

            y_train = y[train_index]
            y_test = y[test_index]

            for j in range(len(embeddings)):
                clf = KNeighborsClassifier(n_neighbors=3)
                X_train = embeddings[j][train_index,:]
                X_test = embeddings[j][test_index,:]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accs[j].append(accuracy_score(y_pred, y_test))
                f1[j].append(f1_score(y_pred, y_test, average='macro'))
        
        for j in range(len(embeddings)):
            avg_accs[j].append(np.mean(accs[j]))
            avg_f1[j].append(np.mean(f1[j]))
            
    accs = list()
    f1 = list()
    for i in range(len(embeddings)):
        accs.append(np.mean(avg_accs[i]))
        f1.append(np.mean(avg_f1[i]))
    
    return accs, f1


def normalizekm(K):
    v = np.sqrt(np.diag(K))
    nm =  np.outer(v,v)
    Knm = np.power(nm, -1)
    Knm = np.nan_to_num(Knm)
    normalized_K = K * Knm
    return normalized_K


def evaluate_graph_classification(K, y):
    # Number of parameter trials
    trials = 8
    n_iters = 10

    # Set the seed for uniform parameter distribution
    random.seed(None) 
    np.random.seed(None) 

    # Number of splits of the data
    splits = 10

    # Normalize kernel matrix
    K = normalizekm(K)
    
    y = y.reshape((-1,1))
    y = np.ravel(y)

    # Size of the dataset
    n = K.shape[0]
    
    # Set up the parameters
    C_grid = 10. ** np.arange(-3,5,1) / n

    correct_pred = []
    val_mean = []
    test_mean = []

    for j in range(n_iters):

        #print("Starting iteration %d..." % (j+1))

        # Initialize the performance of the best parameter trial on validation
        # With the corresponding performance on test
        val_split = []
        test_split = []

        kf = KFold(n_splits=splits, shuffle=True)

        # For each split of the data
        it = 0
        for train_index, test_index in kf.split(K):

            it += 1

            #print("Starting split %d..." % it)

            # Set the training, validation and test
            # Note: the percentage can be set up by the user
            num_train = int((len(train_index) * 90)/100) #90% (of train + val) for training
            num_val = len(train_index) - num_train       # ~10% (of train + val) for validation

            idx = np.random.permutation(len(train_index))
            vtr_idx, vte_idx =  train_index[idx[:num_train]], train_index[idx[num_train:]]

            # Split the kernel matrices
            K_train = K[np.ix_(vtr_idx, vtr_idx)]
            K_val = K[np.ix_(vte_idx, vtr_idx)]
            K_test = K[np.ix_(test_index, vtr_idx)]

            # Split the targets
            y_train = y[vtr_idx]
            y_val = y[vte_idx]
            y_test = y[test_index]

            # Record the performance for each parameter trial
            # respectively on validation and test set
            perf_all_val = []
            perf_all_test = []

            # For each parameter trial
            for i in range(trials):
                # Fit classifier1 on training data
                clf = svm.SVC(kernel = 'precomputed', C = C_grid[i])
                clf.fit(K_train, y_train)

                # predict on validation and test
                y_pred = clf.predict(K_val)

                # accuracy on validation set
                acc = accuracy_score(y_val, y_pred)
                perf_all_val.append(acc)

            # get optimal parameter on validation (argmax accuracy)
            max_idx = np.argmax(perf_all_val)
            C_opt = C_grid[max_idx]

            # performance corresponsing to the optimal parameter on validation
            perf_val_opt = perf_all_val[max_idx]

            clf = svm.SVC(kernel = 'precomputed', C = C_opt)
            clf.fit(K[np.ix_(train_index, train_index)], y[train_index])
            y_pred = clf.predict(K[np.ix_(test_index, train_index)])
            perf_test_opt = accuracy_score(y_test, y_pred)
            for i in range(y_pred.shape[0]):
                if y_pred[i] == y_test[i]:
                    correct_pred.append(1)
                else:
                    correct_pred.append(0)

            #print("\nThe best performance is for trial %d with parameter C = %3f" % (max_idx, C_opt))
            #print("The best performance on the validation set is: %3f" % perf_val_opt)
            #print("The corresponding performance on test set is: %3f" % perf_test_opt)

            # append the best performance on validation
            # at the current split
            val_split.append(perf_val_opt)

            # append the correponding performance on the test set
            test_split.append(perf_test_opt)

        # mean of the validation performances over the splits
        val_mean.append(np.mean(np.asarray(val_split)))
        # std deviation of validation over the splits
        val_std = np.std(np.asarray(val_split))

        # mean of the test performances over the splits
        test_mean.append(np.mean(np.asarray(test_split)))
        # std deviation of the test oer the splits
        test_std = np.std(np.asarray(test_split))

        #print("\nMean performance on val set: %3f" % val_mean[j])
        #print("With standard deviation: %3f" % val_std)
        #print("\nMean performance on test set: %3f" % test_mean[j])
        #print("With standard deviation: %3f" % test_std)

    print("\nMean performance on val set in %d iterations: %3f" % (n_iters, np.mean(np.asarray(val_mean))))
    print("With standard deviation: %3f" % np.std(np.asarray(val_mean)))
    print("\nMean performance on test set in %d iterations: %3f" % (n_iters, np.mean(np.asarray(test_mean))))
    print("With standard deviation: %3f" % np.std(np.asarray(test_mean)))