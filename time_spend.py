import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from models import MLP, CNN,CNN2D,ResNet18,ResNet34,RFSignalCNN


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 25)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(Residual18Block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Residual18Block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _convert_data_size(self, x):
        return x.unsqueeze(1)  # 假设输入的 x 形状是 (batch_size, 102, 63)

    def forward(self, x):
        x = self._convert_data_size(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = F.relu(x)
        features = self.fc1(x)
        d = self.fc2(features)
        outputs = self.fc3(d)
        return features,outputs


class Residual18Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual18Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
from options import args_parser
args = args_parser()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# 加载预训练模型
def load_model(model_path):
    if (args.process == 'RAW_IQ'):
        model = RFSignalCNN(args=args)
    if (args.process == 'stft'):
        model = ResNet18()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 加载数据
def load_data(test_data_path,label_path):
    data = np.load(test_data_path)
    label = np.load(label_path)

    return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

from utils import get_dataset, get_Lora_dataset
from torch.utils.data import DataLoader, Dataset
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]][:-1], self.dataset[self.idxs[item]][-1]
        return torch.tensor(image).float(), torch.tensor(label).long()

class DatasetSplit_stft(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset,label, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.label = label

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]], self.label[self.idxs[item]]
        return torch.tensor(image).float(), torch.tensor(label).long()

# 评估模型并计时
def evaluate_model(args, model, test_dataset):
    loss, total, correct = 0.0, 0.0, 0.0
    start_time = time.time()  # 记录开始时间

    loss_func = nn.CrossEntropyLoss().to(device)
    idx = [i for i in range(len(test_dataset))]
    test_loader = DataLoader(DatasetSplit(test_dataset, idx), batch_size=1,
                             shuffle=True)
    predictions, truths = [], []

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        # Inference
        _, outputs = model(images)
        batch_loss = loss_func(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        predictions.append(pred_labels.cpu().numpy())
        truths.append(labels.cpu().numpy())
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    # 计算平均损失
    loss = loss / len(test_loader)
    accuracy = correct / total

    pred_arr = np.concatenate(predictions).flatten()
    gt_arr = np.concatenate(truths).flatten()
    elapsed_time = time.time() - start_time  # 计算推断时间
    print("设备为：",device)
    print("推断时间：",elapsed_time)

    return accuracy, loss, pred_arr, gt_arr

def evaluate_model_stft(args, model, test_dataset,test_y):
    loss, total, correct = 0.0, 0.0, 0.0
    start_time = time.time()  # 记录开始时间

    loss_func = nn.CrossEntropyLoss().to(device)
    idx = [i for i in range(len(test_dataset))]
    test_loader = DataLoader(DatasetSplit_stft(test_dataset, test_y, idx), batch_size=1,
                             shuffle=True)
    predictions, truths = [], []

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        # Inference
        _, outputs = model(images)
        batch_loss = loss_func(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        predictions.append(pred_labels.cpu().numpy())
        truths.append(labels.cpu().numpy())
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    # 计算平均损失
    loss = loss / len(test_loader)
    accuracy = correct / total

    pred_arr = np.concatenate(predictions).flatten()
    gt_arr = np.concatenate(truths).flatten()
    elapsed_time = time.time() - start_time  # 计算推断时间
    print("设备为：",device)
    print("推断时间：",elapsed_time)

    return accuracy, loss, pred_arr, gt_arr


import torch.quantization as quant

# 准备量化模型
# 训练后量化（Post-Training Quantization）
def post_training_quantize(model):
    # 使用量化配置来量化模型
    model.eval()  # 确保模型在评估模式
    model_q = quant.quantize_dynamic(
        model,  # 要量化的模型
        {nn.Linear},  # 量化层的类型，这里以线性层为例
        dtype=torch.qint8  # 设置量化为 int8 精度
    )
    return model_q

def post_training_quantize_static(model, calibration_data, device):
    # 设置量化配置
    model.eval()  # 确保模型在评估模式
    # 修改量化配置为 per_tensor_affine
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):  # 跳过卷积层
            module.qconfig = None  # 禁用卷积层的量化
        else:
            # 为其他层设置量化配置
            module.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    # 在 calibration 数据集上运行，以便确定激活范围
    calibration_data = torch.tensor(calibration_data, dtype=torch.float32)  # 确保为 float32
    calibration_data = calibration_data.to(device)
    # 运行模型来收集量化信息
    model(calibration_data)
    # 应用量化
    torch.quantization.convert(model, inplace=True)
    return model


# 计算量化前后的参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# 获取模型的内存大小
def get_model_size(model):
    # 获取模型的所有参数
    model_size = sum(param.numel() * param.element_size() for param in model.parameters())
    return model_size



# 主函数
def main():
    print(torch.__version__)  # 检查 PyTorch 版本
    if (args.process == 'RAW_IQ'):
        model_path = 'centralized_weights_WIFI_RFSignalCNN_80.pt'  #
    if (args.process == 'stft'):
        model_path = 'centralized_weights_WIFI_ResNet18_80.pt'
    # 加载模型
    model = load_model(model_path).to(device)

    train_dataset, adapt_dataset, test_dataset, train_y, adapt_y, test_y = get_Lora_dataset(args)
    if (args.process == 'RAW_IQ'):
        te_acc, te_loss, te_pred, te_gt  = evaluate_model(args, model, test_dataset)
    if (args.process == 'stft'):
        te_acc, te_loss, te_pred, te_gt = evaluate_model_stft(args, model, test_dataset, test_y)
    print(f"Final Loss on Test: {te_loss:.4f}")
    print(f"Final Accuracy on Test: {te_acc:.2%}")


    # 生成混淆矩阵
    def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        绘制混淆矩阵
        :param cm: 混淆矩阵
        :param labels: 类别标签
        :param title: 图表标题
        :param cmap: 颜色映射
        """
        plt.figure(figsize=(20, 16))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        # 在每个格子里加入数值
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="red")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def calculate_confusion_matrix(predictions, labels, num_classes):
        """
        计算混淆矩阵
        :param predictions: 模型预测标签
        :param labels: 真实标签
        :param num_classes: 类别数
        :return: 归一化混淆矩阵和原始计数矩阵
        """
        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions, labels=np.arange(num_classes))
        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm_normalized, cm


    cm = confusion_matrix(te_gt, te_pred)
    # 计算混淆矩阵
    num_classes = len(np.unique(te_gt))  # 假设类别是从0到num_classes-1
    print("num_classes:",num_classes)
    cm_norm, cm_counts = calculate_confusion_matrix(te_pred, te_gt, num_classes)
    # 定义类别标签
    class_labels = [f"Class {i}" for i in range(num_classes)]
    # 绘制归一化混淆矩阵
    plot_confusion_matrix(cm_norm, labels=class_labels, title="Normalized Confusion Matrix")

    #量化前精度为torch.float32
    # for param in model.parameters():
    #     print(param.dtype)

    print("量化前的模型参数量:", count_parameters(model))
    model_size_before = get_model_size(model)
    print(f"量化前的模型内存占用: {model_size_before / (1024 ** 2):.2f} MB")
    # 训练后量化
    model = post_training_quantize(model)
    #model = post_training_quantize_static(model, test_dataset[:100], device)
    print("完成量化")
    print("量化后的模型参数量:", count_parameters(model))
    # 量化后的内存占用
    model_size_after = get_model_size(model)
    print(f"量化后的模型内存占用: {model_size_after / (1024 ** 2):.2f} MB")
    #量化后精度为
    for name, param in model.named_parameters():
        print(f"Layer: {name}, dtype: {param.dtype}")





    if (args.process == 'RAW_IQ'):
        te_acc, te_loss, te_pred, te_gt = evaluate_model(args, model, test_dataset)
    if (args.process == 'stft'):
        te_acc, te_loss, te_pred, te_gt = evaluate_model_stft(args, model, test_dataset, test_y)

    print(f"量化后的最终测试损失: {te_loss:.4f}")
    print(f"量化后的最终测试精度: {te_acc:.2%}")
    # 生成混淆矩阵
    cm = confusion_matrix(te_gt, te_pred)
    num_classes = len(np.unique(te_gt))
    print("num_classes:", num_classes)
    cm_norm, cm_counts = calculate_confusion_matrix(te_pred, te_gt, num_classes)
    class_labels = [f"Class {i}" for i in range(num_classes)]
    plot_confusion_matrix(cm_norm, labels=class_labels, title="Normalized Confusion Matrix")


if __name__ == '__main__':
    main()
