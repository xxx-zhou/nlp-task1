import numpy
import csv
import random
from feature import Bag,Gram
from comparison_plot import plot_alpha_gradient
from nlp_task1.Softmax_regression import Softmax
import matplotlib.pyplot as plt

# 数据读取
with open("train.tsv") as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp = list(tsvreader)

# 初始化
data = temp[1:]
max_item=1000
random.seed(2021)
numpy.random.seed(2021)

# 特征提取
bag=Bag(data,max_item)
bag.get_words()
bag.get_matrix()

gram=Gram(data, dimension=2, max_item=max_item)
gram.get_words()
gram.get_matrix()

def regression_gradient_plot(bag, gram, total_times, mini_size, dataset_type):
    """Plot categorization verses different parameters for a given dataset."""
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    shuffle_train, shuffle_test = [], []
    batch_train, batch_test = [], []
    mini_train, mini_test = [], []

    for alpha in alphas:
        soft = Softmax(len(bag.train), 5, bag.len)
        soft.regression(bag.train_matrix, bag.train_y, alpha, total_times, "shuffle")
        r_train, r_test = soft.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
        shuffle_train.append(r_train)
        shuffle_test.append(r_test)

        soft.regression(bag.train_matrix, bag.train_y, alpha, int(total_times / bag.max_item), "batch")
        r_train, r_test = soft.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
        batch_train.append(r_train)
        batch_test.append(r_test)

        soft.regression(bag.train_matrix, bag.train_y, alpha, int(total_times / mini_size), "mini", mini_size)
        r_train, r_test = soft.correct_rate(bag.train_matrix, bag.train_y, bag.test_matrix, bag.test_y)
        mini_train.append(r_train)
        mini_test.append(r_test)
        plt.subplot(2, 2, 1)
        plt.semilogx(alphas, shuffle_train, 'r--', label='shuffle')
        plt.semilogx(alphas, batch_train, 'g--', label='batch')
        plt.semilogx(alphas, mini_train, 'b--', label='mini-batch')
        plt.semilogx(alphas, shuffle_train, 'ro-', alphas, batch_train, 'g+-', alphas, mini_train, 'b^-')
        plt.legend()
        plt.title(f"{dataset_type} -- Training Set")
        plt.xlabel("Learning Rate")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)

        plt.subplot(2, 2, 2)
        plt.semilogx(alphas, shuffle_test, 'r--', label='shuffle')
        plt.semilogx(alphas, batch_test, 'g--', label='batch')
        plt.semilogx(alphas, mini_test, 'b--', label='mini-batch')
        plt.semilogx(alphas, shuffle_test, 'ro-', alphas, batch_test, 'g+-', alphas, mini_test, 'b^-')
        plt.legend()
        plt.title(f"{dataset_type} -- Test Set")
        plt.xlabel("Learning Rate")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)

    def plot_alpha_gradient(bag, gram, total_times, mini_size):
        plt.figure(figsize=(10, 8))
        regression_gradient_plot(bag, gram, total_times, mini_size, "Bag of words")
        plt.subplot(2, 2, 3)
        regression_gradient_plot(gram, bag, total_times, mini_size, "N-gram")
        plt.tight_layout()
        plt.show()

plot_alpha_gradient(bag,gram,10000,10)  # 计算10000次
plot_alpha_gradient(bag,gram,100000,10)  # 计算100000次