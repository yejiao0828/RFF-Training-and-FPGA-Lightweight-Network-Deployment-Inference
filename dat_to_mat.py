
import os
import numpy as np
import struct
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from SpectrogramGenerator import SpectrogramGenerator
from options import args_parser
from scipy.io import loadmat
# 读取并处理 I/Q 数据
import scipy.io as sio

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
        print("iq data shape:",np.array(iq_data['data']).shape)
        return iq_data['data']


# 切片 I/Q 数据
def slice_iq_data(iq_data, slice_size, overlap=0):
    step = slice_size - overlap
    slices = [
        iq_data[i:i + slice_size]
        for i in range(0, len(iq_data) - slice_size + 1, step)
    ]
    return slices


# 保存数据为 .mat 格式
def save_as_mat(data, output_path):
    # 将数据保存为 .mat 格式
    sio.savemat(output_path, {'data': data})

# 示例使用：将处理过的 `.dat` 数据保存为 `.mat` 格式
def process_and_save_mat(base_dir, slice_size, overlap=0):
    all_slices = []
    all_labels = []
    count = 0
    device_path = base_dir
    all_slices = []
    all_labels = []
    count = 0
    output_base_dir = 'C:/Lora_DATA/MAT/Day5'  # 设定保存的文件夹路径
    for label, device_folder in enumerate(sorted(os.listdir(base_dir))):
        device_path = os.path.join(base_dir, device_folder)

        if os.path.isdir(device_path):  # 如果是文件夹
            device_slices = []
            for file_index in range(1, 11):  # 遍历 IQ_1.dat 到 IQ_5.dat
                file_path = os.path.join(device_path, f'IQ_{file_index}.dat')
                if os.path.exists(file_path):  # 如果文件存在
                    print(f"Processing {file_path}")

                    iq_data = read_iq_data(file_path)
                    slices = iq_data

                    device_slices.extend(slices)
                    print("device_slices.shape:", np.array(device_slices).shape)

           # 保存该设备的数据为 .mat 格式
            output_file = os.path.join(output_base_dir, f"{device_folder}_day5.mat")
            save_as_mat(np.array(device_slices), output_file)  # 保存为 mat 文件
            print(f"数据已保存到 {output_file}")


# 调用示例
base_dir = 'C:/Lora_DATA/Day5'  # 数据目录
slice_size = 8192  # 每个切片的样本数
overlap = 0  # 每个切片之间的重叠样本数
process_and_save_mat(base_dir, slice_size, overlap)
