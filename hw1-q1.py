#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        yhat = self.predict(x_i)
        if yhat != y_i:
            self.W[y_i, :] = self.W[y_i, :] + x_i
            self.W[yhat, :] = self.W[yhat, :] - x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        lr = 0.001
        ey = np.zeros(10)
        ey[y_i] = 1.0
        pyx = np.ones(10)
        labels = np.matmul(self.W, x_i.T)
       #for label in range(self.W.shape[0]):
        #    num = np.exp(labels[label])
         #   den = np.sum(np.exp(labels))
          #  pyx[label] = num / den
        num = np.exp(labels)
        den = np.sum(np.exp(labels))
        pyx = num/den
        final = np.reshape((pyx - ey), (-1, 1))
        x_i = np.reshape(x_i, (-1, 1))
        gradient = final @ x_i.T
        self.W = self.W - lr*gradient





class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        '''
        self.b0 = np.zeros((hidden_size, 1))
        self.b1 = np.zeros((n_classes, 1))
        self.w0 = np.random.normal(0.1, 0.1, size=(hidden_size, n_features))
        self.w1 = np.random.normal(0.1, 0.1, size=(n_classes, hidden_size))
        '''
        self.b0 = np.zeros((hidden_size, 1))
        self.b1 = np.zeros((n_classes, 1))
        self.w0 = np.random.normal(0.1, 0.1, size=(hidden_size, n_features))
        self.w1 = np.random.normal(0.1, 0.1, size=(n_classes, hidden_size))

    def predict(self, X):
        #Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        predict = []
        for i in range(X.shape[0]):
            "foward"
            x_i = np.reshape(X[i], (X.shape[1], 1))
            z0 = np.matmul(self.w0, x_i) + self.b0
            h0 = (z0 + np.abs(z0)) /2
            z1 = np.matmul(self.w1, h0) + self.b1
            m = z1.max()
            num = np.exp(z1 - m)
            den = np.sum(np.exp(z1 - m))
            softmax = num/den
            predict.append(np.argmax(softmax))
        return predict


    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        print("acc =",n_correct / n_possible)
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        print("X =", X[0])
        for i in range(X.shape[0]):
            "foward"
            x_i = np.reshape(X[i], (X.shape[1], 1))
            y_i = np.ones((10,1))
            y_i[y[i]] = y[i]
            z0 = np.matmul(self.w0, x_i) + self.b0
            h0 = (z0 + np.abs(z0)) / 2
            z1 = np.matmul(self.w1, h0) + self.b1
            m = z1.max()
            num = np.exp(z1-m)
            den = np.sum(np.exp(z1-m))
            softmax = num/den
            "backpropagation"
            grad_z1 = softmax - y_i
            grad_b1 = grad_z1
            grad_w1 = np.matmul(grad_z1, h0.T)

            #grad_h0 = np.matmul(softmax.T, self.w1).T
            grad_h0 = self.w1.T @  grad_z1
            grad_z0 = np.zeros((grad_h0.shape[0], 1))
            '''
            for k in range(grad_z0.shape[0]):
                if z0[k] > 0:
                    grad_z0[k] = grad_h0[k]
            '''
            g_linha = h0
            g_linha[h0>0] = 1
            g_linha[h0<=0] = 0
            grad_z0 = grad_h0 * g_linha

            grad_w0 = np.matmul(grad_z0, x_i.T)
            grad_b0 = grad_z0
            "Stochastic gradient descent"
            self.w1 = self.w1 - learning_rate * grad_w1
            self.w0 = self.w0 - learning_rate * grad_w0
            self.b1 = self.b1 - learning_rate * grad_b1
            self.b0 = self.b0 - learning_rate * grad_b0







def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
