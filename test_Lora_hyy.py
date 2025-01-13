import os
import numpy as np
import struct
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from SpectrogramGenerator import SpectrogramGenerator
from options import args_parser
from scipy.io import loadmat


# 读取 I/Q 数据
args = args_parser()
def read_iq_data(file_path):
    if args.datasettype == "dat":

        with open(file_path, 'rb') as f:
            data = f.read()
            num_samples = len(data) // 8  # 每两个 Float32 字节是一个 I/Q 样本
            iq_data = np.zeros(num_samples, dtype=complex)
            for i in range(num_samples):
                i_val = struct.unpack('f', data[i * 8:i * 8 + 4])[0]
                q_val = struct.unpack('f', data[i * 8 + 4:i * 8 + 8])[0]
                iq_data[i] = complex(i_val, q_val)
        return iq_data
    else:
        iq_data = loadmat(file_path)
        #print("iq data keys:",iq_data.keys())
        print("iq data shape:",np.array(iq_data['dataset']).shape)
        return iq_data['dataset']


# 切片 I/Q 数据
def slice_iq_data(iq_data, slice_size, overlap=0):
    step = slice_size - overlap
    slices = [
        iq_data[i:i + slice_size]
        for i in range(0, len(iq_data) - slice_size + 1, step)
    ]
    return slices

# 处理所有设备的数据
def process_all_devices(base_dir, slice_size, overlap=0):
    all_slices = []
    all_labels = []
    count = 0
    device_path = base_dir
    if os.path.isdir(device_path):  # 如果是文件夹
        device_slices = []
        for file_index in range(1, 26):  # 遍历 IQ_1.dat 到 IQ_5.dat
            #file_path = os.path.join(device_path, f'IQ_{file_index}.dat')
            file_path = os.path.join(device_path, f'Device{file_index}_day3.mat')
            if os.path.exists(file_path):  # 如果文件存在
                print(f"Processing {file_path}")
                iq_data = read_iq_data(file_path)

                all_slices.extend(iq_data)  # 堆叠切片数据
                print("all_slices", np.array(all_slices).shape)
                all_labels.extend([count] * len(iq_data))  # 堆叠标签
                print("all_labels", np.array(all_labels).shape)
                count += 1
            else:  # 如果文件不存在
                print(f"文件 {file_path} 不存在，跳过此文件。")
                count += 1
                continue  # 跳过当前循环，进入下一个文件处理

    # 使用 np.vstack() 和 np.hstack() 来堆叠数据和标签
    all_slices = np.vstack(all_slices)
    all_labels = np.hstack(all_labels)

    return all_slices, all_labels


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


# 数据分割
def split_dataset(data, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须为1"

    # 首先划分训练集和临时集
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=(1 - train_ratio), random_state=42,shuffle=False
    )

    # 然后划分验证集和测试集
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=(1 - val_ratio_adjusted), random_state=42,shuffle=False
    )

    return train_data, val_data, test_data, train_labels, val_labels, test_labels

# 获取每个标签的数据分布
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



base_dir = r'C:\Lora_DATA\Pilot_data\Day3'  # 数据目录
slice_size = 8192  # 每个切片的样本数
overlap = 0  # 每个切片之间的重叠样本数

# 处理所有设备的数据
all_slices, all_labels = process_all_devices(base_dir, slice_size, overlap)
print("打乱前：")
print("前10个标签和中间10个标签：",all_labels[:10],all_labels[3000:3010])
count_label_samples(all_labels, "train_labels")

# 打乱数据
all_slices, all_labels = shuffle_data(all_slices, all_labels)

# 打乱后的标签分布
print("训练集打乱后：")
print("前10个标签：",all_labels[:10])
print("总集合形状:", all_slices.shape)
print("标签形状:", all_labels.shape)
# 数据分割
train_data, val_data, test_data, train_labels, val_labels, test_labels = split_dataset(
    all_slices, all_labels, train_ratio=0.8, val_ratio=0.05, test_ratio=0.15
)

# 保存数据和标签
output_dir = r'D:\Python_Project\FedRFF-main\Lora_data'  # 保存目录
os.makedirs(output_dir, exist_ok=True)  # 创建保存目录，如果不存在的话


np.save(os.path.join(output_dir, 'train_data_mat_open_D3.npy'), train_data)
np.save(os.path.join(output_dir, 'val_data_mat_open_D3.npy'), val_data)
np.save(os.path.join(output_dir, 'test_data_mat_open_D3.npy'), test_data)
np.save(os.path.join(output_dir, 'train_labels_mat_open_D3.npy'), train_labels)
np.save(os.path.join(output_dir, 'val_labels_mat_open_D3.npy'), val_labels)
np.save(os.path.join(output_dir, 'test_labels_mat_open_D3.npy'), test_labels)


print("训练集形状:", train_data.shape)
print("验证集形状:", val_data.shape)
print("测试集形状:", test_data.shape)
print("训练集标签形状:", train_labels.shape)
print("验证集标签形状:", val_labels.shape)
print("测试集标签形状:", test_labels.shape)
print("数据和标签已保存。")
