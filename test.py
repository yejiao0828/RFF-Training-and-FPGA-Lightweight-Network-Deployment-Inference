import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_signal_data(bin_path, json_path):
    """
    从二进制和 JSON 文件中加载数据
    """
    # 读取二进制数据
    with open(bin_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)

    # 读取 JSON 文件
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    # 提取协议信息
    protocol = metadata.get('annotations', {}).get('core:protocol', 'unknown')

    # 提取数据类型
    datatype = metadata.get('global', {}).get('core:datatype', 'unknown')

    day = metadata.get('captures', {}).get('core:day', '')  # 获取天数信息
    # 提取样本数量（用于验证数据完整性）
    sample_count = int(metadata.get('annotations', {}).get('core:sample_count', 0))

    # 提取标签（transmitter location）
    label = metadata.get('annotations', {}).get('transmitter', {}).get('core:location', 'unknown')
    if label == "bes":
        label_number = 0
    elif label == "browning":
        label_number = 1
    elif label == "honors":
        label_number = 2
    else:
        label_number = 3
    return data, protocol, datatype, sample_count, day, label_number


def get_dataset(data_dir):
    """
    筛选WIFI数据，并切分为长度为512的序列
    """
    wifi_day1_sequences = []  # 存储切分后的序列
    wifi_day2_sequences = []  # 存储切分后的序列
    labels_day1 = []  # 存储每个数据的标签
    labels_day2 = []  # 存储每个数据的标签

    for root, _, files in os.walk(data_dir):
        # 找到所有 .bin 文件及其对应的 .json 文件
        bin_files = [f for f in files if f.endswith('.bin')]
        for bin_file in bin_files:
            bin_path = os.path.join(root, bin_file)
            json_path = bin_path.replace('.bin', '.json')  # 对应的 JSON 文件
            if os.path.exists(json_path):  # 确保 JSON 文件存在
                data, protocol, datatype, sample_count, day, label = load_signal_data(bin_path, json_path)
                if protocol == "802.11a":  # 筛选WIFI数据
                    if day == "1":
                        print(f"WIFI数据文件: {bin_file}, 数据类型: {datatype}, 样本数量: {sample_count}, 天数: {day}, 标签: {label}")

                        # 验证数据完整性
                        if len(data) != sample_count * 2:  # cf32 数据每个样本占 2 个 float
                            print(f"数据长度错误，跳过文件: {bin_file}")
                            continue

                        # 转换数据为复数
                        if datatype == "cf32":
                            data = data.view(np.complex64)

                        # 切片为长度为1024的序列
                        num_sequences = len(data) // 1024
                        print("每一个样本可以分成：", num_sequences)
                        sliced_data1 = data[:num_sequences * 1024].reshape(-1, 1024)
                        wifi_day1_sequences.append(sliced_data1)
                        labels_day1.extend([label] * num_sequences)  # 每个切分的数据都会有相同的标签
                    if day == "2":
                        print(f"WIFI数据文件: {bin_file}, 数据类型: {datatype}, 样本数量: {sample_count}, 天数: {day}, 标签: {label}")

                        # 验证数据完整性
                        if len(data) != sample_count * 2:  # cf32 数据每个样本占 2 个 float
                            print(f"数据长度错误，跳过文件: {bin_file}")
                            continue

                        # 转换数据为复数
                        if datatype == "cf32":
                            data = data.view(np.complex64)

                        # 切片为长度为1024的序列
                        num_sequences = len(data) // 1024
                        print("每一个样本可以分成：", num_sequences)
                        sliced_data2 = data[:num_sequences * 1024].reshape(-1, 1024)
                        wifi_day2_sequences.append(sliced_data2)
                        labels_day2.extend([label] * num_sequences)  # 每个切分的数据都会有相同的标签

    # 合并所有序列
    if wifi_day1_sequences:
        wifi_day1_sequences = np.vstack(wifi_day1_sequences)  # 将列表中的所有小数组合并为一个大数组
    # 合并所有序列
    if wifi_day2_sequences:
        wifi_day2_sequences = np.vstack(wifi_day2_sequences)  # 将列表中的所有小数组合并为一个大数组
    return wifi_day1_sequences, wifi_day2_sequences, labels_day1, labels_day2


def split_dataset(wifi_sequences, labels, train_ratio, val_ratio, test_ratio):
    """
    将数据集分为训练、验证和测试集
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "比例之和必须为1"

    # 首先划分为训练集和剩余集
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        wifi_sequences, labels, test_size=(1 - train_ratio), random_state=42
    )

    # 然后将剩余集划分为验证集和测试集
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)  # 调整比例
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=(1 - val_ratio_adjusted), random_state=42
    )

    return train_data, val_data, test_data, train_labels, val_labels, test_labels


def save_to_gzip(data, file_path):
    """
    将数据保存为压缩文件
    """
    # 将 NumPy 数组转换为 Pandas DataFrame
    df = pd.DataFrame(data)
    # 保存为 gzip 格式
    df.to_parquet(file_path, compression='gzip')
    print(f"数据已保存到: {file_path}")


# 主函数
data_dir = '../neu_m046tb444/GlobecomPOWDER/'  # 数据根目录
wifi_day1_sequences, wifi_day2_sequences, labels_day1, labels_day2 =  get_dataset(data_dir)
# wifi_sequences, labels = get_dataset(data_dir)

# 分割数据集
train_data_day1, val_data_day1, test_data_day1, train_labels_day1, val_labels_day1, test_labels_day1 = split_dataset(
    wifi_day1_sequences, labels_day1, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
)
train_data_day2, val_data_day2, test_data_day2, train_labels_day2, val_labels_day2, test_labels_day2 = split_dataset(
    wifi_day2_sequences, labels_day2, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
)
train_data_day2_part = train_data_day2[:100]
train_labels_day2_part = train_labels_day2[:100]
unique_labels, counts = np.unique(train_labels_day2, return_counts=True)
# 打印统计结果
for label, count in zip(unique_labels, counts):
    print(f"Label {label}: {count} instances")


# 保存为压缩文件
output_dir = './output_data/'  # 保存目录
os.makedirs(output_dir, exist_ok=True)

print("训练集day1形状:", train_data_day1.shape)
print("验证集day1形状:", val_data_day1.shape)
print("测试集day1形状:", test_data_day1.shape)
print("训练集day2形状:", train_data_day2.shape)
print("验证集day2形状:", val_data_day2.shape)
print("测试集day2形状:", test_data_day2.shape)

# 保存数据和标签
np.save('./output_data/day1_train.npy', train_data_day1)  # 80% 用于训练
np.save('./output_data/day1_val.npy', val_data_day1)  # 10% 用于验证
np.save('./output_data/day1_test.npy', test_data_day1)  # 10% 用于测试
np.save('./output_data/day2_train.npy', train_data_day2)  # 80%个样本的一部分用于迁移学习adapt
np.save('./output_data/day2_val.npy', val_data_day2)  # 10% 用于验证
np.save('./output_data/day2_test.npy', test_data_day2)  # 10% 用于测试信道影响

np.save('./output_data/day1train_labels.npy', train_labels_day1)  # 保存训练标签
np.save('./output_data/day1val_labels.npy', val_labels_day1)  # 保存验证标签
np.save('./output_data/day1test_labels.npy', test_labels_day1)  # 保存测试标签
np.save('./output_data/day2train_labels.npy', train_labels_day2)  # 保存训练标签
np.save('./output_data/day2val_labels.npy', val_labels_day2)  # 保存验证标签
np.save('./output_data/day2test_labels.npy', test_labels_day2)  # 保存测试标签

# 合并 day1 和 day2 的训练数据
adapta_data = np.vstack((train_data_day2_part, train_data_day1))
adapta_labels = np.hstack((train_labels_day2_part, train_labels_day1))  # 标签也需要合并
print("adapt集形状:", adapta_data.shape)
print("adapt集标签形状:", adapta_labels.shape)
np.save('./output_data/adapta_data.npy', adapta_data)
np.save('./output_data/adapta_labels.npy', adapta_labels)



print("数据保存完成!")





