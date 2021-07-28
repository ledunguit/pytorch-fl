# Hiển thị kết quả, biểu đồ và tiến trình
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dataset
from datasets import get_dataset

# Tham số truyền vào
from params import args_parser

# Torch framework
import torch
from torch.utils.data import DataLoader
from update import test_inference

# Model
from models import CNNMnist
from models import CNNCifar
from models import CNNFashion_Mnist
from models import MLP


if __name__ == '__main__':
    device = 'cpu'
    # Kiểm tra xem có GPU hay không, nếu có thì dùng.
    args = args_parser()
    if args.device:
        if args.device == 'cuda':
            if torch.cuda.is_available():
                device = 'cuda'
                torch.cuda.set_device('cuda')
    train_dataset, test_dataset, _ = get_dataset(args)

    # Xây dựng model
    if args.model == 'cnn':
        # CNN
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fashion-mnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Lỗi: Không có model được truyền vào!')

    # Nạp dữ liệu cho device
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Hàm optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr, weight_decay=1e-4)

    data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print("Đang train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1, batch_idx * len(images), len(data_loader.dataset), 100. * batch_idx / len(data_loader), loss.item()
                ))
            batch_loss.append(loss.item())

        loss_average = sum(batch_loss)/len(batch_loss)
        print('\nTraining loss:', loss_average)
        epoch_loss.append(loss_average)

    # Lưu lại hình ảnh từng epoch và loss của từng epoch
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('EPOCHS')
    plt.ylabel('Training loss')
    plt.savefig("save/baseline_{}_{}_{}.png".format(args.dataset, args.model, args.epochs))

    # Test model
    test_accuracy, test_loss = test_inference(args, global_model, test_dataset)
    print('Test trên', len(test_dataset), 'mẫu')
    print('Độ chính xác: {:.2f}%'.format(100*test_accuracy))

