from subprocess import call
import argparse


def gamma(gpu_idx):
    for gamma in [0.8, 0.9, 0.95, 0.999]:
        call(f'python main.py -g {gpu_idx} -ga {gamma} -c gamma{gamma}'.split())


def replay(gpu_idx):
    for replay in [500000, 250000, 125000, 62500]:
        print(f'running with replay of {replay}')
        call(f'python main.py -g {gpu_idx} -ga 0.999 -r {replay} -c replay{replay}'.split())


# def lr(gpu_idx): # don't run this
#     for lr in [25e-4]:  # 25e-4 is the default
#         call(f'python main.py -g {gpu_idx} -ga 0.999 -l {lr} -c lr{lr}'.split())


def softmax_beta(gpu_idx, single=None):
    beta_list = [0.1, 1, 10, 100]
    if single is not None:
        beta_list = [beta_list[single]]
    for beta in beta_list:
        call(f'python main.py -g {gpu_idx} -ga 0.999 -r 125000 -a softmax -b {beta} --bsched True -c beta{beta}_bsched'.split())


def noisy(gpu_idx, single=None):
    todo = None
    s_list  = [10, 1, 1e-1, 1e-2]
    if single is not None:
        s_list = [s_list[single]]
    for istd in s_list:
        call(f'python main.py -g {gpu_idx} -ga 0.999 -r 125000 -a noisy -i {istd} -c istd{istd}'.split())


def main():
    parser = argparse.ArgumentParser(description='Hyperparamater tuning')
    parser.add_argument('-g', '--gpu_idx', type=int, default=0)
    parser.add_argument('-s', '--single', type=int)
    parser.add_argument('-m', '--method', type=str)
    args = parser.parse_args()
    if args.method == 'softmax':
        softmax_beta(args.gpu_idx, args.single)
    elif args.method == 'noisy':
        noisy(args.gpu_idx, args.single)
    elif args.method == 'gamma':
        gamma(args.gpu_idx, args.single)
    elif args.method == 'reply':
        replay(args.gpu_idx, args.single)




if __name__ == '__main__':
    main()
