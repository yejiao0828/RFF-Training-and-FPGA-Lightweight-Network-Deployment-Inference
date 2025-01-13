# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 15:02
# @Author       : tl22089
# @File         : utils.py
# @Affiliation  : University of Bristol
import copy
from collections import deque
import numpy as np
#import torch
import pandas as pd
from numpy.random import standard_normal, uniform
from SpectrogramGenerator import SpectrogramGenerator

def count_label_samples(labels, dataset_name):
    """
    Count the number of samples for each label in the dataset.

    Args:
        labels (numpy.ndarray): Array of labels.
        dataset_name (str): Name of the dataset (e.g., "train", "adapt", "test").

    Returns:
        None
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution in {dataset_name}:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} samples")

def gaussian_noise(data, snr_range):
    arr = np.zeros(data.shape, dtype=complex)
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0], snr_range[-1], pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        SNR_linear = 10 ** (SNRdB[pktIdx] / 10)
        P = sum(abs(s) ** 2) / len(s)
        N0 = P / SNR_linear
        n = np.sqrt(N0 / 2) * (standard_normal(len(s)) + 1j * standard_normal(len(s)))
        arr[pktIdx] = s + n

    return arr


def add_noise(data, args, flag):
    # data_real = data.iloc[:, :1024]
    # data_imag = data.iloc[:, 1024:]
    data_real = np.real(data)
    data_imag = np.imag(data)
    arr = data

    # n = len(data_real)
    # arr = np.zeros(data_real.shape, dtype=complex)
    # for idx in range(n):
    #     dp = data_real.iloc[idx].values + 1j * data_imag.iloc[idx].values
    #     # print(dp)
    #     arr[idx] = dp
    if args.noise and flag:
        arr = gaussian_noise(arr, snr_range=(args.snr_low, args.snr_high))
    else:
        print("不加噪")
    c1 = arr.real
    c2 = arr.imag
    c = np.concatenate([c1, c2], axis=1)
    return c

def add_noise_stft(data, args, flag):

    arr = data
    if args.noise and flag:
        arr = gaussian_noise(arr, snr_range=(args.snr_low, args.snr_high))
    else:
        print("不加噪处理")
    return arr


def data_iid(data, num_users):
    num_items = int(len(data)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(data))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def filter_signal(data, sig):
    data = data[data['radio'] == sig]
    bs_code = data[['radio', 'radio_code', 'bs', 'bs_code']].drop_duplicates()
    print(bs_code)
    data_sig = data.drop(['radio', 'radio_code', 'bs'], axis=1)
    return data_sig

def count_label_samples(labels, dataset_name):
    """
    Count the number of samples for each label in the dataset.

    Args:
        labels (numpy.ndarray): Array of labels.
        dataset_name (str): Name of the dataset (e.g., "train", "adapt", "test").

    Returns:
        None
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution in {dataset_name}:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} samples")
def shuffle_data(all_slices, all_labels):
    """
    打乱数据和标签，但保持对应关系不变。

    参数：
    - all_slices: np.array, 包含所有切片的数据。
    - all_labels: np.array, 包含所有切片的标签。

    返回：
    - shuffled_slices: np.array, 打乱后的切片数据。
    - shuffled_labels: np.array, 打乱后的标签数据。
    """
    # 将数据和标签组合成一个二维数组
    combined = list(zip(all_slices, all_labels))

    # 打乱数据和标签的顺序
    np.random.shuffle(combined)

    # 拆分回切片和标签
    shuffled_slices, shuffled_labels = zip(*combined)

    # 转换回 numpy 数组
    shuffled_slices = np.array(shuffled_slices)
    shuffled_labels = np.array(shuffled_labels)

    return shuffled_slices, shuffled_labels

def generate_spectrogram(data):
    """
    生成时频图
    """
    spectrogram_generator = SpectrogramGenerator()

    # 生成时频图
    spectrograms = []
    for sample in data:
        #print("sample shape",np.array(sample).shape)
        sample = sample[np.newaxis, :]
        #print("sample shape", np.array(sample).shape)
        spectrogram = spectrogram_generator.channel_ind_spectrogram(sample)  # 每个样本生成时频图
        spectrograms.append(spectrogram)

    return np.array(spectrograms)

def auto_concatenate_samples(data, group_size, target_length):
    """
    自动按组拼接样本，每组包含 group_size 个样本。
    """
    concatenated_signals = []
    num_samples = data.shape[0]

    # 按组拼接样本
    for i in range(0, num_samples, group_size):
        group_samples = data[i:i + group_size]  # 获取一组样本
        concatenated_signal = []

        # 拼接当前组的样本
        for sample in group_samples:
            concatenated_signal.extend(sample)

        # 如果拼接后的信号长度超过目标长度，截断
        if len(concatenated_signal) > target_length:
            concatenated_signal = concatenated_signal[:target_length]

        # 如果拼接后的信号长度不足目标长度，补零
        elif len(concatenated_signal) < target_length:
            concatenated_signal.extend([0] * (target_length - len(concatenated_signal)))

        concatenated_signals.append(np.array(concatenated_signal))

    return np.array(concatenated_signals)


def get_Lora_dataset(args):
    #加载保存的 .npy 数据
    data_dir = '../Lora_data/output/'
    print("***当前的训练数据集和测试数据集来源：***：",args.datasettype)
    print("***当前是RAW I/Q还是STFT：", args.process)
    print("***当前的训练数据集和测试数据集是否是同一天：***：", args.data_choice)
    if args.data_choice == "Trainday1_Testday1":
        if args.datasettype == "dat":
            train_x = np.load(data_dir + 'train_data_dat.npy')
            adapt_x = np.load(data_dir + 'val_data_dat.npy')
            test_x = np.load(data_dir + 'test_data_dat.npy')
            train_y = np.load(data_dir + 'train_labels_dat.npy')
            adapt_y = np.load(data_dir + 'val_labels_dat.npy')
            test_y = np.load(data_dir + 'test_labels_dat.npy')
        elif args.datasettype == "mat":
            train_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_data_mat_open_D1.npy')
            adapt_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\val_data_mat_open_D1.npy')
            test_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_data_mat_open_D1.npy')
            train_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_labels_mat_open_D1.npy')
            adapt_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\val_labels_mat_open_D1.npy')
            test_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_labels_mat_open_D1.npy')
        else:
            print(f"数据读取错误，未成功导入数据")
    elif args.data_choice == "Trainday1_Testday3":
        if args.datasettype == "dat":
            train_x = np.load(data_dir + 'train_data_dat.npy')
            adapt_x = np.load(data_dir + 'val_data_dat.npy')
            test_x = np.load(data_dir + 'train_data_dat_D3.npy')
            test_x = test_x[13000:15000]
            train_y = np.load(data_dir + 'train_labels_dat.npy')
            adapt_y = np.load(data_dir + 'val_labels_dat.npy')
            test_y = np.load(data_dir + 'train_labels_dat_D3.npy')
            test_y = test_y[13000:15000]
        elif args.datasettype == "mat":
            train_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_data_mat_open_D1.npy')
            adapt_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\val_data_mat_open_D3.npy')
            test_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_data_mat_open_D3.npy')
            train_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_labels_mat_open_D1.npy')
            adapt_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\val_labels_mat_open_D3.npy')
            test_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_labels_mat_open_D3.npy')
            print("检查训练样本是否对齐")
            print(train_x[0])
            print(test_x[0])
        else:
            print(f"数据读取错误，未成功导入数据")
    elif args.data_choice == "Trainadapt_Testday3":
        train_x_D1 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_data_mat_open_D1.npy')
        train_y_D1 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_labels_mat_open_D1.npy')
        train_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_data_mat_open_D3.npy')
        adapt_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\val_data_mat_open_D3.npy')
        test_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_data_mat_open_D3.npy')
        train_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_labels_mat_open_D3.npy')
        adapt_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\val_labels_mat_open_D3.npy')
        test_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_labels_mat_open_D3.npy')
        # Concatenate the training data and labels
        # count_label_samples(train_y[:5000],'train_y[:5000]')
        train_x = np.concatenate((train_x_D1, train_x), axis=0)
        train_y = np.concatenate((train_y_D1, train_y), axis=0)
        # print("检查训练样本是否对齐")
        # print(train_x[0])
        # print(train_x_D1[0])
    elif args.data_choice == "Trainall_Testday3":
        # 加载数据
        train_x_D1 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_data_mat_open_D1.npy')
        train_y_D1 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_labels_mat_open_D1.npy')
        train_x_D2 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_data_mat_open_D2.npy')
        train_y_D2 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_labels_mat_open_D2.npy')
        train_x_D4 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_data_mat_open_D4.npy')
        train_y_D4 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_labels_mat_open_D4.npy')
        train_x_D5 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_data_mat_open_D5.npy')
        train_y_D5 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_labels_mat_open_D5.npy')
        train_x_D3 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_data_mat_open_D3.npy')
        train_y_D3 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_labels_mat_open_D3.npy')

        adapt_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\val_data_mat_open_D3.npy')
        adapt_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\val_labels_mat_open_D3.npy')

        test_x_D1 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_data_mat_open_D1.npy')
        test_y_D1 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_labels_mat_open_D1.npy')
        test_x_D2 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_data_mat_open_D2.npy')
        test_y_D2 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_labels_mat_open_D2.npy')
        test_x_D3 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_data_mat_open_D3.npy')
        test_y_D3 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_labels_mat_open_D3.npy')
        test_x_D4 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_data_mat_open_D4.npy')
        test_y_D4 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_labels_mat_open_D4.npy')
        test_x_D5 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_data_mat_open_D5.npy')
        test_y_D5 = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_labels_mat_open_D5.npy')

        # 输出各个数据集的形状，检查它们是否一致
        print("train_x_D1 形状:", train_x_D1.shape)
        print("train_x_D2 形状:", train_x_D2.shape)
        print("train_x_D3 形状:", train_x_D3.shape)
        print("train_x_D4 形状:", train_x_D4.shape)
        # print("train_x_D5 形状:", train_x_D5.shape)
        # 拼接数据
        train_x = np.concatenate((train_x_D1, train_x_D2, train_x_D3, train_x_D4, train_x_D5), axis=0)
        train_y = np.concatenate((train_y_D1, train_y_D2, train_y_D3, train_y_D4, train_y_D5), axis=0)
        # train_x = train_x_D1[:2500]  #100，200，400样本数用这两行代码
        # train_y = train_y_D1[:2500]

        # test_x = np.concatenate((test_x_D1, test_x_D2, test_x_D3, test_x_D4, test_x_D5), axis=0)
        # test_y = np.concatenate((test_y_D1, test_y_D2, test_y_D3, test_y_D4, test_y_D5), axis=0)
        test_x = test_x_D5
        test_y = test_y_D5
        print("打乱数据")
        train_x, train_y = shuffle_data(train_x, train_y)
        test_x, test_y = shuffle_data(test_x, test_y)
        # train_x = train_x[:10000]
        # train_y = train_y[:10000]
        test_x  = test_x[:2000]
        test_y  = test_y[:2000]

        count_label_samples(train_y, "train_y")
        count_label_samples(test_y, "test_y")
        # 输出拼接后的数据形状，确保它们已按顺序拼接
        print("拼接后的 train_x 形状:", train_x.shape)
        print("拼接后的 train_y 形状:", train_y.shape)
        print("拼接后的 testx 形状:", test_x.shape)
        print("拼接后的 test_y 形状:", test_y.shape)

    elif args.data_choice == "Trainday3_Testday3":
        train_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_data_mat_open_D3.npy')
        adapt_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\val_data_mat_open_D3.npy')
        test_x = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_data_mat_open_D3.npy')
        train_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\train_labels_mat_open_D3.npy')
        adapt_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\val_labels_mat_open_D3.npy')
        test_y = np.load(r'D:\Python_Project\FedRFF-main\Lora_data\test_labels_mat_open_D3.npy')

    # print("加噪前样本的量纲：", train_x[0])
    print("train_samples.shape:",train_x.shape)
    print("adapt_samples.shape:",adapt_x.shape)
    print("test_samples.shape:",test_x.shape)
    print("train_y.shape:",train_y.shape)
    print("adapt_y.shape:",adapt_y.shape)
    print("test_y.shape:",test_y.shape)

    if(args.process == 'stft'):
        #加噪
        train_x = add_noise_stft(train_x, args, flag=True)
        adapt_x = add_noise_stft(adapt_x, args, flag=True)
        test_x = add_noise_stft(test_x, args, flag=True)
        print("加噪后的train_samples.shape:",train_x.shape)
        print("加噪后的adapt_samples.shape:",adapt_x.shape)
        print("加噪后的test_samples.shape:",test_x.shape)
        # 生成时频图特征
        train_spectrograms = generate_spectrogram(train_x)
        adapt_spectrograms = generate_spectrogram(adapt_x)
        test_spectrograms = generate_spectrogram(test_x)
        print("去除维度前train_spectrograms.shape: ",train_spectrograms.shape)
        train_spectrograms = np.squeeze(train_spectrograms)
        adapt_spectrograms = np.squeeze(adapt_spectrograms)
        test_spectrograms = np.squeeze(test_spectrograms)

        #print(train_spectrograms[0][0])
        print("生成时频图后的train_spectrograms.shape:", train_spectrograms.shape)
        print("生成时频图后的adapt_spectrograms.shape:", adapt_spectrograms.shape)
        print("生成时频图后的test_spectrograms.shape:", test_spectrograms.shape)
        # Save spectrograms to .npy files
        # 将数据和标签保存在一个字典中
        test_data = {'spectrograms': test_spectrograms, 'labels': test_y}
        # 保存数据和标签到一个 .npy 文件
        np.save('test_spectrograms.npy', test_data)
        print("时频图已保存为test_spectrograms.npy")

        return train_spectrograms, adapt_spectrograms, test_spectrograms,train_y,adapt_y,test_y
    if (args.process == 'RAW_IQ'):
        train_samples = add_noise(train_x, args, flag=True)
        adapt_samples = add_noise(adapt_x, args, flag=True)
        test_samples = add_noise(test_x, args, flag=True)
        print("加噪后的train_samples.shape:",train_samples.shape)
        print("加噪后的adapt_samples.shape:",adapt_samples.shape)
        print("加噪后的test_samples.shape:",test_samples.shape)
        print("标准化前样本的量纲：", train_samples[0])
        #标准化操作
        train_samples = train_samples.astype('float32')

        mean = train_samples.ravel().mean()
        std = train_samples.ravel().std()

        train_samples = (train_samples - mean) / std
        adapt_samples = (adapt_samples - mean) / std
        test_samples = (test_samples - mean) / std
        # #将标签加到数据的最后一列
        train_samples = np.append(train_samples, train_y[:, np.newaxis], axis=1)
        adapt_samples = np.append(adapt_samples, adapt_y[:, np.newaxis], axis=1)
        test_samples = np.append(test_samples, test_y[:, np.newaxis], axis=1)
        print("标准化后检查样本的量纲：", train_samples[0])
        print("加标签后的train_samples.shape:",train_samples.shape)
        print("加标签后的adapt_samples.shape:",adapt_samples.shape)
        print("加标签后的test_samples.shape:",test_samples.shape)
        # 将数据和标签保存在一个字典中
        test_data = {'RAWIQ': test_samples, 'labels': test_y}
        # 保存数据和标签到一个 .npy 文件
        np.save('test_RAWIQ.npy', test_data)
        train_groups = data_iid(train_samples, args.num_users)
        adapt_groups = data_iid(adapt_samples, args.num_users)

        return train_samples, adapt_samples, test_samples,train_y,adapt_y,test_y


# def average_weights(w):
#     """
#     Returns the average of the weights.
#     """
#     w_avg = copy.deepcopy(w[0])
#     for key in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[key] += w[i][key]
#         w_avg[key] = torch.div(w_avg[key], len(w))
#     return w_avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {}  # save all data values here
        self.save_dict = {}  # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count


