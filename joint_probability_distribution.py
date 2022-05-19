import numpy as np
from scipy.stats.contingency import margins


def entropy(array):
    """
    Calculates the entropy of a discrete random variable
    :param array: probabilities of the discrete random variable
    :return: Entropy of the discrete random variable described by array
    """
    return -1 * np.sum(array * np.log2(array))


def relative_entropy(p, q):
    """
    Calculates the Kullback-Leibler distance between two discrete random variables p, q D(p||q)
    :param p: discrete random variable
    :param q: discrete random variable
    :return: Kullback-Leibler distance D(p||q)
    """
    logarithms = np.log2(p / q, out=np.zeros_like(p), where=(p != 0))
    return np.sum(np.multiply(p, logarithms))


class JointProbabilityDistribution:
    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def all(self):
        marginal_distributions = self.marginal_distributions
        conditional_entropy = self.conditional_entropy
        return {
            'marginal distribution x': marginal_distributions[0],
            'marginal distribution y': marginal_distributions[1],
            'joined entropy': self.joined_entropy,
            'marginal entropy x': entropy(marginal_distributions[0]),
            'marginal entropy y': entropy(marginal_distributions[1]),
            'conditional entropy x|y': conditional_entropy[0],
            'conditional entropy y|x': conditional_entropy[1],
            'relative entropy x||y': self.relative_entropy('x'),
            'relative entropy y||x': self.relative_entropy('y'),
            'joined distribution': self.matrix
        }

    @property
    def marginal_distributions(self):
        x, y = margins(self.matrix)
        return x.T.flatten(), y.flatten()

    @property
    def joined_entropy(self):
        logarithms = np.log2(self.matrix, out=np.zeros_like(self.matrix), where=(self.matrix != 0))
        products = np.multiply(self.matrix, logarithms)
        return -1 * np.sum(products)

    @property
    def conditional_entropy(self):
        return self.__conditional_entropy(axis=0), self.__conditional_entropy(axis=1)

    def relative_entropy(self, first='x'):
        marginal_distributions = self.marginal_distributions

        if first == 'x':
            p, q = marginal_distributions
        elif first == 'y':
            q, p = marginal_distributions
        else:
            raise ValueError('Permitted values: {x,y}')
        return relative_entropy(p, q)

    @property
    def mutual_information(self):
        x, y = self.marginal_distributions
        p = self.matrix
        q = np.multiply(x, y)
        return relative_entropy(p, q)

    def __conditional_entropy(self, axis=0):
        axis_a, axis_b = (0, 1) if axis == 0 else (1, 0)

        p_b = self.marginal_distributions[axis_a]

        result = 0

        for i in range(self.matrix.shape[axis_a]):
            for j in range(self.matrix.shape[axis_b]):
                if self.matrix[i, j] == 0:
                    continue
                p_xy = self.matrix[i, j]
                p_a_given_b = p_xy / p_b[j]
                result += p_xy * np.log2(p_a_given_b)

        return -1 * result


if __name__ == '__main__':
    matrix = np.matrix([
        [1 / 8, 1 / 16, 1 / 32, 1 / 32],
        [1 / 16, 1 / 8, 1 / 32, 1 / 32],
        [1 / 16, 1 / 16, 1 / 16, 1 / 16],
        [1 / 4, 0, 0, 0],
    ])
    jointProbabilityDistribution = JointProbabilityDistribution(matrix)

    for key, value in jointProbabilityDistribution.all.items():
        print(f'{key}: {value}')
