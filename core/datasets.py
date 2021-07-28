from make_datasets import make_cifar_iid
from make_datasets import make_cifar_noniid
from make_datasets import make_fashion_mnist_iid
from make_datasets import make_fashion_mnist_noniid
from make_datasets import make_fashion_mnist_noniid_unequal
from make_datasets import make_mnist_iid
from make_datasets import make_mnist_noniid
from make_datasets import make_mnist_noniid_unequal

from torchvision import datasets, transforms


def get_dataset(args):

    if args.dataset == 'mnist':
        save_dir = './data/'
        transform_apply = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


        # Fix lỗi server không hoạt động, không tải được dataset
        datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]

        train_dataset = datasets.MNIST(
            save_dir, train=True, download=True, transform=transform_apply
        )

        test_dataset = datasets.MNIST(
            save_dir, train=False, download=True, transform=transform_apply
        )



        if args.is_iid:
            user_groups = make_mnist_iid(train_dataset, args.num_users)
        else:
            if args.is_unequal:
                user_groups = make_mnist_noniid_unequal(
                    train_dataset, args.num_users)
            else:
                user_groups = make_mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'fashion-mnist':
        save_dir = './data/'
        transform_apply = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.FashionMNIST(
            save_dir, train=True, download=True, transform=transform_apply
        )

        test_dataset = datasets.FashionMNIST(
            save_dir, train=False, download=True, transform=transform_apply
        )

        if args.is_iid:
            user_groups = make_fashion_mnist_iid(train_dataset, args.num_users)
        else:
            if args.is_unequal:
                user_groups = make_fashion_mnist_noniid_unequal(
                    train_dataset, args.num_users
                )
            else:
                user_groups = make_fashion_mnist_noniid(
                    train_dataset, args.num_users)

    elif args.dataset == 'cifar':
        save_dir = './data/'
        transform_apply = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.CIFAR10(
            save_dir, train=True, download=True, transform=transform_apply)

        test_dataset = datasets.CIFAR10(
            save_dir, train=False, download=True, transform=transform_apply)

        if args.is_iid:
            user_groups = make_cifar_iid(train_dataset, args.num_users)
        else:
            if args.is_unequal:
                raise NotImplementedError()
            else:
                user_groups = make_cifar_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups
