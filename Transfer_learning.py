import torch
from models import CNN  # 确保模型结构和模块路径一致
from resnet import ResNet18
import numpy as np
# 设置设备
device = 'cpu'

# 加载预训练模型（完整模型）
global_model = torch.load('../save/CNNmodel/centralized_model_signal_model_snr.pth')
global_model.to(device)
global_model.eval()

data_dir = '../src/output_data/'
adapt_x = np.load(data_dir +'day2_adapt.npy.npy')
adapt_y = np.load(data_dir +'day2adapt_labels.npy')
# adapt_x = adapt_x[:1000]
# adapt_y = adapt_y[:1000]

