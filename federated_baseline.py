# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 20:34
# @Author       : tl22089
# @File         : federated_baseline.py
# @Affiliation  : University of Bristol
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import datetime
import torch
import numpy as np
import random
import os
import copy
import time
import pickle
from torch.utils.tensorboard import SummaryWriter
from utils import get_dataset, average_weights,get_Lora_dataset
from options import args_parser
from update import test_inference, LocalUpdate
from update_TripletCNN import test_inference_T, LocalUpdate_T
from models import MLP, CNN,CNN2D,ResNet18,ResNet34,RFSignalCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    args.type = 'non-metric'
    seed = 2022
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 将设备设置为GPU，如果GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # print(device)
    # define paths
    now = datetime.datetime.now()
    log_name = '../logs/Fed-baseline-{:}-{:}-{:}-{:}-{:}-{:}-model-{:}-iid-{:}-bs-{:}/'.format(now.year, now.month,
                                                                                               now.day,
                                                                                               now.hour,
                                                                                               now.minute,
                                                                                               now.second,
                                                                                               args.model,
                                                                                               args.iid,
                                                                                               args.bs_classes)
    logger = SummaryWriter(log_name)
    # load datasets
    train_dataset, adapt_dataset, test_dataset, train_y,adapt_y, test_y = get_Lora_dataset(args)
    print("***完成数据导入、预处理***")

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        global_model = CNN(args=args)
    elif args.model == 'cnn2D':
        global_model = CNN2D(args=args)
    elif args.model == 'RFSignalCNN':
        global_model = RFSignalCNN(args=args)
    elif args.model == 'ResNet18':
        global_model = ResNet18(args=args)
    elif args.model == 'ResNet34':
        global_model = ResNet34(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        global_model = MLP(args=args)
    else:
        exit('Error: unrecognized model')        # Convolutional neural network

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    train_bs_acc, train_radio_acc = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    # 创建两个新列表来记录测试集的损失和准确率
    test_loss_list, test_accuracy_list = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0


    for round in range(args.epochs):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        count = 0
        for idx in idxs_users:
            count = count + 1
            print(f"第{count}个客户端")
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=train_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=round)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)  #local_losses是每个round中所有客户端的损失值数组
        train_loss.append(loss_avg)                       #train_loss是每个round的平均损失
        logger.add_scalar('train/loss', loss_avg, round)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss, bs_acc, radio_acc = [], [], [], []
        global_model.eval()
        for c in range(args.num_users):
            if args.Base_or_Tri == 'BaselineCNN':
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=train_groups[c], logger=logger)
            elif args.Base_or_Tri == 'TripletCNN':
                local_model = LocalUpdate_T(args=args, dataset=train_dataset,
                                          idxs=train_groups[c], logger=logger)
            else:
                print("******error with args.Base_or_Tri******")

            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
            print(f"客户端 {c}, 精度: {acc:.4f}")  # 打印每个客户端的准确率
        train_accuracy.append(sum(list_acc) / len(list_acc))
        logger.add_scalar('train/acc', train_accuracy[-1], round)

        # print global training loss after every 'i' rounds
        print(f' \nTraining Stats after {round + 1} global rounds:')
        print(f'Current round Training Loss : {loss_avg:.4f}')
        print('|---- Train Accuracy: {:.2f}%\n'.format(100. * train_accuracy[-1]))

        # Test inference after Current round Training
        test_acc, test_loss, _ , _  = test_inference(args, global_model, test_dataset)
        print("|---- Test Accuracy: {:.2f}%".format(100. * test_acc))
        test_loss_list.append(test_loss)  # 保存测试集损失
        test_accuracy_list.append(test_acc)  # 保存测试集准确率

    # 保存完整模型（结构+权重）
    model_path = '../save/Federated_CNNmodel/fed_baseline_{:}.pth'.format(args.model)
    torch.save(global_model, model_path)

    # 保存权重（仅参数）
    weights_path = '../save/Federated_CNNmodel/fed_baseline_{:}.pt'.format(args.model)
    torch.save(global_model.state_dict(), weights_path)

    # Test inference after completion of training
    test_acc, test_loss,te_pred, te_gt = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Final Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|----  Final Test Accuracy: {:.2f}%".format(100. * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/Federated_CNNmodel/fed_baseline_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # 保存训练和测试损失、准确率到文本文件
    def save_to_txt(file_name, train_loss, train_accuracy, test_loss_list, test_accuracy_list):
        with open(file_name, 'w') as f:
            f.write('Round Train_Loss Train_Accuracy Test_Loss Test_Accuracy\n')  # 写入标题行
            for i in range(len(train_loss)):
                # 写入每一轮的数据
                f.write(
                    f'{i + 1} {train_loss[i]:.4f} {train_accuracy[i]:.4f} {test_loss_list[i]:.4f} {test_accuracy_list[i]:.4f}\n')

    # 定义保存的文件路径
    file_name = '../save/Federated_CNNmodel/loss_accuracy_data_C[{}]_BT[{}]_B[{}]_L[{}]_rounds[{}].txt'.format(
        args.data_choice, args.Base_or_Tri, args.local_bs, args.local_ep, args.epochs)
    # 在训练结束后保存数据
    save_to_txt(file_name, train_loss, train_accuracy, test_loss_list, test_accuracy_list)
    print(f'Loss and accuracy data saved to {file_name}')

    # 生成混淆矩阵
    cm = confusion_matrix(te_gt, te_pred)
    # 计算混淆矩阵的概率分布
    row_sums = cm.sum(axis=1, keepdims=True)  # 每个类别的真实标签总数
    cm_prob = cm / row_sums  # 归一化为概率分布
    # 类别标签（假设类别标签是从 0 到 n_classes - 1 的整数）
    class_labels = [str(i) for i in range(cm.shape[0])]
    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_prob, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.title('RFF_Confusion_Matrix_Probabilities')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('../save/CNNmodel/confusion_matrix_probs_{:}_{:}_{:}.png'.format(args.signal, args.model, args.snr_high))
    plt.show()

    # 绘制测试集损失曲线
    plt.figure()
    plt.title('Train & Test Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r', label='Train Loss')
    plt.plot(range(len(test_loss_list)), test_loss_list, color='b', label='Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Communication Rounds')
    plt.legend()
    plt.show()

    # 绘制测试集准确率曲线
    plt.figure()
    plt.title('Train & Test Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k', label='Train Accuracy')
    plt.plot(range(len(test_accuracy_list)), test_accuracy_list, color='g', label='Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Communication Rounds')
    plt.legend()
    plt.show()



