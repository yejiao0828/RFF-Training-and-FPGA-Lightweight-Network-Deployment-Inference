import numpy as np
import matplotlib.pyplot as plt
from SpectrogramGenerator import SpectrogramGenerator


# 加载训练集数据
def load_train_data(file_path):
    """
    加载训练数据
    """
    train_data = np.load(file_path + '/train_data_mat_open_D1.npy')
    train_data_D3 = np.load(file_path + '/train_data_mat_open_D3.npy')
    train_labels = np.load(file_path + '/train_labels_mat_open_D1.npy')
    train_labels_D3 = np.load(file_path + '/train_labels_mat_open_D3.npy')

    return train_data, train_labels, train_data_D3, train_labels_D3


# 使用 STFT 生成时频图
def generate_spectrogram(data):
    """
    生成单个信号的时频图
    """
    spectrogram_generator = SpectrogramGenerator()
    spectrogram = spectrogram_generator.channel_ind_spectrogram(data)  # 数据传入单个样本
    return spectrogram


# 自动每隔 8 个样本拼接
def auto_concatenate_samples(data, group_size, target_length):
    """
    自动按组拼接样本，每组包含 group_size 个样本。
    """
    concatenated_signals = []
    num_samples = data.shape[0]

    # 按组拼接样本
    for i in range(10000, num_samples, group_size):
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


# 示例用法
if __name__ == "__main__":
    # 设置数据路径
    output_dir = 'D:\Python_Project\FedRFF-main\Lora_data'  # 修改为你的训练数据路径

    # 加载训练数据
    train_data, train_labels, train_data_D3, train_labels_D3 = load_train_data(output_dir)

    # 打印数据形状
    print(f"训练集数据形状: {train_data.shape}")
    print(f"训练集数据形状: {train_data_D3.shape}")
    print(f"训练集标签形状: {train_labels_D3.shape}")

    sample_index = 1  # 选择第100个样本
    print("label:", train_labels[sample_index])
    data = train_data_D3[sample_index][np.newaxis, :]  # 形状为(1, 8192)
    print(np.array(data).shape)

    # 计算信号幅值
    magnitude_data = np.abs(data[0])  # 提取幅值

    # 可视化信号的幅值
    plt.plot(magnitude_data)
    plt.title(f"Amplitude of Signal at Device {train_labels[sample_index]}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()
