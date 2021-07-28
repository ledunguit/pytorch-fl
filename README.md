# Xây dựng mô hình Federated Learning sử dụng framework Pytorch

### Các bài báo tham khảo

* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)

Các dataset sử dụng: MNIST, FMNIST, CIFAR10

Các model sử dụng: MLP, CNN

## Thư viện yêu cầu
```bash
virtualenv my_env
source my_env/bin/active
```

## Dataset
Dataset có thể được tải thủ công vào đường dẫn data/ hoặc được tải tự động dựa vào tham số --dataset, toàn bộ dataset đều được tải từ torchvision

## Thực nghiệm
# Huấn luyện mô hình theo cách truyền thống:
Sử dụng CNN:
```bash
python core/basic_main.py --model=cnn --dataset=mnist --epochs=50
```

Nếu có GPU
```bash
python core/basic_main.py --model=cnn --dataset=mnist --device=cuda --epochs=50
```

Sử dụng MLP:
```bash
python core/basic_main.py --model=mlp --dataset=mnist --epochs=50
```

Nếu có GPU
```bash
python core/basic_main.py --model=mlp --dataset=mnist --device=cuda --epochs=50
```

# Huấn luyện mô hình theo mô hình học hợp tác:

Sử dụng CNN:
```bash
python core/federated_main.py --model=cnn --dataset=mnist --epochs=50
```

Nếu có GPU
```bash
python core/federated_main.py --model=cnn --dataset=mnist --device=cuda --epochs=50
```

Sử dụng MLP:
```bash
python core/federated_main.py --model=mlp --dataset=mnist --epochs=50
```

Nếu có GPU
```bash
python core/federated_main.py --model=mlp --dataset=mnist --device=cuda --epochs=50
```

## Các tham số khác:

* ```--dataset:  Mặc định: 'mnist'. Tùy chọn: 'mnist', 'fmnist', 'cifar' ```
* ```--model:    Mặc định: 'mlp'. Options: 'mlp', 'cnn' ```
* ```--device:      Mặc định: Chạy bằng CPU. Hoặc chạy bằng GPU nếu set tham số cuda. ```
* ```--epochs:   Số vòng huấn luyện. ```
* ```--lr:       Learning rate. Mặc định là 0.1 ```
* ```--verbose:  Log chi tiết. Mặc định bật, để tắt đưa vào giá trị 0 ```

## Các tham số dành cho học hợp tác
* ```--is_iid:      Cách phân phối dữ liệu cho các người dùng tham gia. Mặc định I.I.D. Nhập 0 cho Non-I.I.D. ```
* ```--num_users: Số lượng user tham gia. Mặc định là 100. ```
* ```--frac:     Tỉ lệ số lượng người dùng sẽ được lấy model update mỗi round để tổng hợp vào global model. ```
* ```--local_ep: Số epochs huấn luyện cho từng người dùng. ```
* ```--local_bs: Batch size của local model của người dùng. ```
* ```--is_unequal:  Nếu cách phân phối dữ liệu là non_iid thì tham số này sẽ quyết định dữ liệu được chia đều cho người dùng hay không. ```