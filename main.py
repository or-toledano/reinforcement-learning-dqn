import gym
import torch.optim as optim
import argparse
import warnings
from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01


def main(task, seed, num_timesteps):
    parser = argparse.ArgumentParser(description='DQN on visual atari board')
    parser.add_argument('-g', '--gpu_idx', type=int, default=0)
    parser.add_argument('-ga', '--gamma', type=float, default=GAMMA)
    parser.add_argument('-l', '--learning', type=float, default=LEARNING_RATE)
    parser.add_argument('-lo', '--loss', default='MSE')
    parser.add_argument('-c', '--config_name', default='')
    parser.add_argument('-s', '--stats_name')
    parser.add_argument('-d', '--dir')
    parser.add_argument('-a', '--actor', default='greedy', help='greedy, softmax or noisy')
    parser.add_argument('-b', '--beta', type=float, default=1, help='softmax beta')
    parser.add_argument('--bsched', type=bool, default=False, help='beta scheduling')
    parser.add_argument('-i', '--init_std', type=float, default=1e-1, help='Initial std for noisy actor')
    parser.add_argument('-r', '--replay_buffer_size', type=int, default=REPLAY_BUFFER_SIZE)
    args = parser.parse_args()
    gpu_idx = args.gpu_idx

    if args.dir:
        direc = args.dir
        warnings.warn(
            "args.dir is deprecated, use config_name instead",
            DeprecationWarning
        )
    else:
        direc = 'tmp/gym-results/'
        if args.config_name:
            direc = f'{direc}{args.config_name}/'

    if args.stats_name:
        stats_name = args.stats_name
        warnings.warn(
            "args.stats_name is deprecated, use config_name instead",
            DeprecationWarning
        )
    else:
        stats_name = 'statistics.pkl'
        if args.config_name:
            stats_name = f'statistics_{args.config_name}.pkl'

    env = get_env(task, seed, direc, video_callable=False)

    def stopping_criterion(env_arg):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env_arg, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=args.learning, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=args.replay_buffer_size,
        batch_size=BATCH_SIZE,
        gamma=args.gamma,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
        gpu_idx=gpu_idx,
        loss=args.loss,
        stat_name=stats_name,
        actor_name=args.actor,
        beta=args.beta,
        init_std=args.init_std,
        bsched=args.bsched
    )


if __name__ == '__main__':
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    # bm_task = benchmark.tasks[3]  # pong
    bm_task = benchmark.tasks[1]  # breakout

    # Run training
    main_seed = 0  # Use a seed of zero (you may want to randomize the seed!)

    main(bm_task, main_seed, bm_task.max_timesteps)
