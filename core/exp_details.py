
def exp_details(args):
    print('\nChi tiết thực nghiệm:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning rate : {args.lr}')
    print(f'    Tổng round global   : {args.epochs}\n')

    print('    Các tham số học hợp tác:')
    if args.is_iid:
        print('    IID data được chọn')
    else:
        print('    Non-IID data được chọn')
    print(f'    Tỉ lệ người dùng  : {args.frac}')
    print(f'    Batch size tại local   : {args.local_bs}')
    print(f'    Số epochs tại local       : {args.local_ep}\n')
    return