import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    return parser.parse_args()
