# coding:utf-8
import numpy as np
import re
import itertools
from collections import Counter
import importlib, sys
importlib.reload(sys)

# 剔除英文的符号
def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(clear_data_file, misl_data_file):
    """
    加载二分类训练数据，为数据打上标签
    (X,[0,0])
    X = 内容
    Y = 标签

    0:诱导---> [1,0]

    1:非诱导--->[0,1]

    (X,Y)

    """

    clear_examples = list(open(clear_data_file, "r", encoding='utf-8').readlines())
    clear_examples = [s.strip() for s in clear_examples]  #去除空格？好像不用
    misl_examples = list(open(misl_data_file, "r", encoding='utf-8').readlines())
    misl_examples = [s.strip() for s in misl_examples]
    x_text = clear_examples + misl_examples
    # 适用于英文
    # x_text = [clean_str(sent) for sent in x_text]

    x_text = [sent for sent in x_text]
    # 定义类别标签 ，格式为one-hot的形式: y=1--->[0,1]
    clear_labels = [[0, 1] for _ in clear_examples]
    # print positive_labels[1:3]
    misl_labels = [[1, 0] for _ in misl_examples]
    y = np.concatenate([clear_labels, misl_labels], 0)
    """
    print y
    [[0 1]
     [0 1]
     [0 1]
     ...,
     [1 0]
     [1 0]
     [1 0]]
    print y.shape
    (10662, 2)
    """
    return [x_text, y]

# 就是dataloader（相当于dataloader的init函数）
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    利用迭代器从训练数据的回去某一个batch的数据
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # 每回合打乱顺序
        if shuffle:
            # 随机产生以一个乱序数组，作为数据集数组的下标
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        # 划分批次
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# 测试代码用的
if __name__ == '__main__':
    clear_data_file = './fenci/clear.txt'
    misl_data_file = './fenci/mislead.txt'
    load_data_and_labels(clear_data_file, misl_data_file)









