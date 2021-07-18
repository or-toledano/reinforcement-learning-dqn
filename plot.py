import pickle
import matplotlib
from matplotlib import pyplot as plt
from glob import glob
from argparse import ArgumentParser
matplotlib.use('Agg')


def plot(param: str, out: str):
    files = glob(f'statistics_{param}*.pkl')  # ~4 files will take ~7.5GB of RAM
    for file in files:
        with open(file, 'rb') as f:
            print('f: ', f)
            stats = pickle.load(f)
        mean = stats['mean_episode_rewards']
        best_mean = stats["best_mean_episode_rewards"]
        timesteps = len(mean)
        plt.plot(range(timesteps), mean, label=f'Mean episode reward for {file}')
        plt.plot(range(timesteps), best_mean, label=f'Best mean episode reward for {file}')
    plt.legend(loc='lower right', fontsize='xx-small')
    plt.title('100 Episode Mean Reward Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Reward')
    plt.savefig(out)


def main():
    #param = 'beta'
    # param = 'lr'
    #param = 'replay'
    #param = 'gamma'
    #param = 'beta'
    #param = 'istd'
    parser = ArgumentParser()
    parser.add_argument('-p', '--param', type=str)
    args = parser.parse_args()
    param = args.param
    plot(param, f'q2_{param}_breakout.png')


if __name__ == '__main__':
    main()
