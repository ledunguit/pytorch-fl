from exp_details import exp_details
from average import average_weights
from datasets import get_dataset
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from update import LocalUpdate, test_inference
from params import args_parser
from tensorboardX import SummaryWriter
import torch
import os
import copy
import time
import pickle
import numpy as np
from torch import cuda
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('logs')

    args = args_parser()
    exp_details(args, 'fl')
    device = "cpu"
    if args.device:
        if args.device == "cuda":
            if torch.cuda.is_available():
                torch.cuda.set_device("cuda")
                device = "cuda"
            else:
                exit('Lỗi: Không tìm thấy phần cứng GPU hỗ trợ, vui lòng kiểm tra lại.')

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # CNN
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer Preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Lỗi: Không có model được truyền vào!')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(f'Chi tiết model:',args.model)
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # tính global weights
        global_weights = average_weights(local_weights)

        # cập nhật global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Tính độ chính xác trung bình của toàn bộ user sau mỗi round
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # In ra kết quả sau từng round
        if (epoch + 1) % print_every == 0:
            print(f' \nSố liệu trung bình sau {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Độ chính xác: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Kết quả sau {args.epochs} global rounds training:')
    print(
        "|---- Độ chính xác trung bình: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test độ chính xác dự đoán: {:.2f}%".format(100*test_acc))

    torch.save(global_model.state_dict(
    ), "save/models/{}_{}_{}.pth".format(args.dataset, args.model, args.epochs))

    # Lưu objects train_loss và train_accuracy:
    file_name = 'save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.is_iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Tổng thời gian: {0:0.4f}'.format(time.time()-start_time))

    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.is_iid, args.local_ep, args.local_bs))

    # Lưu hình ảnh cho độ chính xác trung bình và các round
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.is_iid, args.local_ep, args.local_bs))
