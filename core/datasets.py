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
        save_dir = 'data/mnist'
        transform_apply = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

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
        save_dir = 'data/fashion-mnist'
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
        save_dir = 'data/cifar'
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
