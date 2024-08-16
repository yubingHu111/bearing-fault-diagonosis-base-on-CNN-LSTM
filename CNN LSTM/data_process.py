import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from joblib import dump

torch.manual_seed(1)

def DataReading(file_path, file_name):
    """
    数据读取
    :param file_path: 子文件路径
    :param file_name: 各个文件名--list
    :return: data_set
    """
    data_set = pd.DataFrame()
    index = 0
    for name in file_name:
        data_path = os.path.join(file_path, name)
        dataset = pd.read_csv(data_path, header=None)
        dataset.columns = [index]
        dataset = dataset[0:500499]
        data_set = pd.concat([data_set, dataset], axis=1)
        index = index + 1
    return data_set


def ShowSignal(data_signal, fs):
    """
    signal图形化
    :param data_signal: 所有振动信号（dataframe）
    :param fs: 采样频率
    :return:
    """

    s_p = data_signal.shape[0]
    s_t = s_p / fs
    X = np.linspace(0, s_t, s_p)
    num_signal = data_signal.shape[1]
    plt.figure(figsize=(15, 2 * num_signal))

    for i in range(num_signal):
        plt.subplot(num_signal, 1, i + 1)
        plt.plot(X, data_signal[i])
        plt.title(str(i))

    plt.tight_layout()
    plt.show()


def DataSplit(data, label, window, overlap_rate):
    """
    重新采样，对数据进行拆分
    :param label: 数据标签（int）
    :param data: 数据（一维数组）
    :param window: 一个样本的长度（int）--1024
    :param overlap_rate: 重叠率（int)--0.5
    :return:data_split -- 重新采样的信号（sample,window+1）
    """
    stride = int(window * (1 - overlap_rate))  # 计算步幅
    sample = (len(data) - window) // stride + 1  # 样本数

    data_split = pd.DataFrame(columns=[x for x in range(window + 1)])
    data_list = []
    for i in range(sample):
        sta_index = stride * i
        end_index = window + sta_index
        temp_data = data[sta_index:end_index].tolist()
        temp_data.append(label)
        data_list.append(temp_data)

    data_split = pd.DataFrame(data_list, columns=[data_split.columns])
    return data_split


def Make_data(file_name, file_path, num_class, window, overlap_rate, split_scale):
    """
    数据集制作
    :param file_name: 文件名（list）
    :param file_path: 文件夹名（str)
    :param num_class: 分类数量
    :param window: 每个样本长度
    :param overlap_rate: 重叠率
    :param split_scale: 训练测试数据集划分比例
    :return:train_set,test_set（dataframe)(样本数，单个样本长度）
    """
    data_set = DataReading(file_path, file_name)

    # 重新采样数据
    labels = [x for x in range(num_class)]  # 十二分类
    resample_data = pd.DataFrame()
    for index in range(len(labels)):
        data = data_set[index]
        data_split = DataSplit(data, index, window, overlap_rate)
        resample_data = pd.concat([resample_data, data_split], axis=0)  # resample(8376,1025)样本数，单个样本长度

    resample_data = resample_data.sample(frac=1).reset_index(drop=True)  # 打乱样本顺序

    # 划分数据集
    train_len = int(resample_data.shape[0] * (1 - split_scale))
    train_set = resample_data.iloc[0:train_len, :]
    test_set = resample_data.iloc[train_len:, :]
    train_sample_num = train_set.shape[0]
    test_sample_num = test_set.shape[0]
    return train_set, test_set, train_sample_num, test_sample_num


def Make_Data_Label(data_set):
    X_data = data_set.iloc[:, 0:-1]  # 振动信号值
    Y_labels = data_set.iloc[:, -1]  # 故障标签
    x_data = torch.tensor(X_data.values).float()  # 将pd中的value取出，转换成tensor类型的float
    y_labels = torch.tensor(Y_labels.values.astype("int64"))

    return x_data, y_labels


if __name__ == "__main__":
    file_name = ["ib600_2.csv", "ib800_2.csv", "ib1000_2.csv", "n600_2.csv", "n800_2.csv", "n1000_2.csv",
                 "ob600_2.csv", "ob800_2.csv", "ob1000_2.csv", "tb600_2.csv", "tb800_2.csv", "tb1000_2.csv"]
    file_path = r"D:\document_s\vs_code\数据集\JNU-bearing-dataset\Raw data (原始数据)"

    num_class = 12
    window = 1024
    overlap_rate = 0.3
    split_scale = 0.2

    # 划分数据集--训练样本、测试样本
    train_set, test_set, train_num, test_num = Make_data(file_name, file_path, num_class, window, overlap_rate,
                                                         split_scale)
    print(train_num, test_num)

    dump(train_set, "train_set_pd")  # 保存数据集
    dump(test_set, "test_set_pd")

    # 制作tensor格式的数据集
    train_x_data, train_y_labels = Make_Data_Label(train_set)
    test_x_data, test_y_labels = Make_Data_Label(test_set)
    # 保存tensor格式数据集
    dump(train_x_data, "train_x_tensor")
    dump(train_y_labels, "train_y_tensor")
    dump(test_x_data, "test_x_tensor")
    dump(test_y_labels, "test_y_tensor")

