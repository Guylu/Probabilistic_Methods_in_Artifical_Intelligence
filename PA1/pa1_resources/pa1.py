"""
67800 - Probabilistic Methods in AI
Spring 2021
Programming Assignment 1 - Bayesian Networks
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.io import loadmat
from scipy.special import logsumexp


def get_p_z1(z1_val):
    """
    Get the prior probability for variable Z1 to take value z1_val.
    """
    return bayes_net['prior_z1'][z1_val]


def get_p_z2(z2_val):
    """
    Get the prior probability for variable Z2 to take value z2_val.
    """
    return bayes_net['prior_z2'][z2_val]


def get_p_x_cond_z1_z2(z1_val, z2_val):
    """
    Get the conditional probabilities of variables X_1 to X_784 to take the value 1 given z1 = z1_val and z2 = z2_val.
    """
    return bayes_net['cond_likelihood'][(z1_val, z2_val)]


def get_pixels_sampled_from_p_x_joint_z1_z2():
    """
    return the sampled values of pixel variables (array of length 784)
    """
    # uni choice:
    # ran = np.arange(-3, 3.25, 0.25)
    # z1, z2 = np.random.choice(ran), np.random.choice(ran)
    # res =  get_p_x_cond_z1_z2(z1, z2)
    get_probs = lambda get_p, vals: np.random.choice(vals, p=np.vectorize(get_p)(vals))
    return np.random.binomial(1, get_p_x_cond_z1_z2(get_probs(get_p_z1, z1_vals), get_probs(get_p_z2, z2_vals)))


def get_expectation_x_cond_z1_z2(z1_val, z2_val):
    """
    calculate E(X1:784 | z1, z2): expectation of X1...784 w.r.t the conditional probability
    """
    return get_p_x_cond_z1_z2(z1_val, z2_val)


def get_conditional_expectation(data):
    """
    Calculate the conditional expectation E((z1, z2) | X = data[i]) for each data point
    :param data: Row vectors of data points X (n x 784)
    :return: array of E(z1 | X = data), array of E(z2 | X = data)
    """
    sumz1, sumz2 = np.zeros((1, data.shape[0])), np.zeros((1, data.shape[0]))
    logged = get_log_p_x(data)
    for i, z1_val in enumerate(z1_vals):
        for j, z2_val in enumerate(z2_vals):
            prob = np.exp(get_log_p_x_joint_z1_z2(data, z1_val, z2_val) - logged)
            sumz1 += prob * z1_val
            sumz2 += prob * z2_val
    return sumz1.flatten(), sumz2.flatten()


def get_log_p_x(data):
    """
    Compute the marginal log likelihood: log P(X)
    :param data: Row vectors of data points X (n x 784)
    :return: Array of log-likelihood values
    """
    data_log_likelihood = -np.inf
    for i, z1_val in enumerate(z1_vals):
        for j, z2_val in enumerate(z2_vals):
            # print(i, j)
            data_log_likelihood = np.logaddexp(get_log_p_x_joint_z1_z2(data, z1_val, z2_val),
                                               data_log_likelihood)

    return data_log_likelihood


def get_log_p_x_joint_z1_z2(data, z1_val, z2_val):
    """
    Compute the joint log probability log P(X, z1, z2)
    :param data: Row vectors of data points X (n x 784)
    :param z1_val: z1 value (scalar)
    :param z2_val: z2 value (scalar)
    :return: Array of log probability values
    """
    summ = np.zeros((1, data.shape[0]))
    p = np.log(get_p_x_cond_z1_z2(z1_val, z2_val))
    for i, x in enumerate(data):
        ret = np.zeros((1, x.shape[0]))
        x = x.reshape(1, -1)
        ret[x == 1] = p[x == 1]
        ret[x == 0] = np.log(1 - np.exp(p[x == 0]))
        summ[:, i] = np.sum(ret) + get_p_z1(z1_val) + get_p_z2(z2_val)

    return summ


def q_1():
    """
    Plots the pixel variables sampled from the joint distribution as 28 x 28 images.
    Your job is to implement get_pixels_sampled_from_p_x_joint_z1_z2.
    """
    plt.figure()
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(get_pixels_sampled_from_p_x_joint_z1_z2().reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('q_4', bbox_inches='tight')
    plt.show()
    plt.close()


def q_2():
    """
    Plots the expected images for each latent configuration on a 2D grid.
    Your job is to implement get_expectation_x_cond_z1_z2.
    """

    canvas = np.empty((28 * len(z1_vals), 28 * len(z1_vals)))
    for i, z1_val in enumerate(z1_vals):
        for j, z2_val in enumerate(z2_vals):
            canvas[(len(z1_vals) - i - 1) * 28:(len(z1_vals) - i) * 28, j * 28:(j + 1) * 28] = \
                get_expectation_x_cond_z1_z2(z1_val, z2_val).reshape(28, 28)

    plt.figure()
    plt.axis('off')
    plt.imshow(canvas, cmap='gray')
    plt.tight_layout()
    plt.savefig('q_2', bbox_inches='tight')
    plt.show()
    plt.close()


def q_3():
    """
    Loads the data and plots the histograms. Rest is TODO.
    Your job is to compute real_marginal_log_likelihood and corrupt_marginal_log_likelihood below.
    """

    mat = loadmat('q_3.mat')
    val_data = mat['val_x']
    test_data = mat['test_x']

    '''
    TODO. Calculate marginal_log_likelihood of test samples classified as real and as corrupted.
    '''
    # Your code should calculate the two arrays below...
    val = get_log_p_x(val_data)
    val_mean = np.mean(val)
    val_std = np.std(val)
    print("all mean and std:")
    print(val_mean)
    print(val_std)

    test_log = get_log_p_x(test_data)
    real_marginal_log_likelihood = test_log[test_log > val_mean - 3 * val_std]

    corrupt_marginal_log_likelihood = test_log[test_log < val_mean - 3 * val_std]

    plot_histogram(real_marginal_log_likelihood,
                   title='Histogram of marginal log-likelihood for real test data',
                   xlabel='marginal log-likelihood', savefile='q_3_hist_real')

    plot_histogram(corrupt_marginal_log_likelihood,
                   title='Histogram of marginal log-likelihood for corrupted test data',
                   xlabel='marginal log-likelihood', savefile='q_3_hist_corrupt')

    plt.show()
    plt.close()


def q_4():
    """
    Loads the data and plots a color coded clustering of the conditional expectations.
    Your job is to implement the get_conditional_expectation function
    """

    mat = loadmat('q_4.mat')
    data = mat['x']
    labels = mat['y']

    mean_z1, mean_z2 = get_conditional_expectation(data)

    plt.figure()
    plt.scatter(mean_z1, mean_z2, c=np.squeeze(labels))
    plt.colorbar()
    plt.grid()
    plt.savefig('q_4', bbox_inches='tight')
    plt.show()
    plt.close()


def load_model(model_file):
    """
    Loads a default Bayesian network with latent variables (in this case, a variational autoencoder)
    """

    with open(model_file + '.pkl', 'rb') as infile:
        cpts = pkl.load(infile, encoding='bytes')

    model = {}
    model['prior_z1'] = cpts[0]
    model['prior_z2'] = cpts[1]
    model['cond_likelihood'] = cpts[2]

    return model


def plot_histogram(data, title='histogram', xlabel='value', ylabel='frequency', savefile='hist'):
    """
    Plots a histogram.
    """

    plt.figure()
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')


def main():
    global bayes_net, z1_vals, z2_vals
    bayes_net = load_model('trained_mnist_model')
    z1_vals = sorted(bayes_net['prior_z1'].keys())
    z2_vals = sorted(bayes_net['prior_z2'].keys())

    q_1()
    q_2()
    q_3()
    q_4()


if __name__ == '__main__':
    main()

# 1. 2^784 -1
# 2. 2^786 -1
