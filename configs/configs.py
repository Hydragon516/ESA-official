# DATA
DATA = {
    'dataset': 'CULane',
    'data_root': '/HDD1/mvpservereight/minhyeok/Lane_Detection/CUlane/list',
    'num_class': 4
}

# TRAIN
TRAIN = {
    'device': '0, 1',
    'epoch': 45,
    'train_batch_size': 28,
    'valid_batch_size': 10,
    'num_workers': 16,
    'learning_rate': 0.1,
    'step': 15,
    'print_freq': 100,
    'eval_freq': 1
}

# TEST
TEST = {
    'device': '0, 1',
    'checkpoint': './trained/ep005.pth'
}