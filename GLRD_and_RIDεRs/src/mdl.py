import huffman
import numpy as np
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq


class MDL:
    def __init__(self, bins):
        self.threshold = 1e-5
        self.code_bins = bins

    def code_frequencies(self, matrix):
        frequencies = {}
        for (x, y), code in np.ndenumerate(matrix):
            if x == y:
                continue
            if code in frequencies:
                frequencies[code] += 1.0
            else:
                frequencies[code] = 1.0
        return frequencies

    def get_huffman_code_length(self, sub_matrix, return_code='avg'):
        # return_code 'avg': returns the average code length per symbol in bit (default)
        # otherwise: returns the total symbol length in bit required to encode the data

        whitened_mat = whiten(sub_matrix)
        #print(whitened_mat.shape, self.code_bins)
        code_book, distortion = kmeans(whitened_mat, self.code_bins, thresh=self.threshold)
        quantized_mat = vq(whitened_mat, code_book)
        frequencies = self.code_frequencies(quantized_mat)
        frequency_values = [frequencies[code] for code in frequencies.keys()]
        Z = sum(frequency_values)
        probabilities = [x / Z for x in frequency_values]
        huffman_codes = huffman.huffman(probabilities)

        if return_code == 'avg':
            return huffman.symbol_code_expected_length(probabilities, huffman_codes)  # Avg. symbol bit len
        else:
            return sum(value * len(code) for value, code in zip(frequency_values, huffman_codes))  # total sym bits

    def get_reconstruction_error(self, actual_matrix, estimated_matrix):
        # KLD based reconstruction error.
        # For more details please refer:
        # Algorithms for Non-negative Matrix Factorization by Lee and Seung, NIPS 2001.

        reconstruction_error = 0.0
        for (i, j), value in np.ndenumerate(actual_matrix):
            if i == j:
                continue
            if actual_matrix[i][j] < 1e-10 or estimated_matrix[i][j] < 1e-10:
                continue
            reconstruction_error += (actual_matrix[i][j] * np.log2(actual_matrix[i][j] / estimated_matrix[i][j])
                                         - actual_matrix[i][j] + estimated_matrix[i][j])
        return reconstruction_error

    def get_log_likelihood(self, actual_matrix, estimated_matrix):
        a = list(actual_matrix.flatten())
        E = np.abs(actual_matrix-estimated_matrix)
        err = list(E.flatten())
        return -0.5 * np.log2(np.exp(1)) / (np.var(a)) * sum([val**2 for val in err])