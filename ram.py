import gym
import torch.optim as optim
import argparse

from dqn_model import DQN_RAM
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_ram_env, get_wrapper_by_name
from utils.schedule import LinearSchedule

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 1
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01


def main(env, num_timesteps, args):
    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=DQN_RAM,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
        gpu_idx=args.gpu_idx,
        cfg_name=args.cfg_name
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN on ram atari')
    parser.add_argument('-g', '--gpu_idx', type=int, default=0)
    parser.add_argument('-ga', '--gamma', type=float, default=0.99)
    parser.add_argument('-l', '--learning', type=float, default=0.00025)
    parser.add_argument('-cn', '--cfg_name', type=str, default='', help='name for result files, e.g. gamma095')
    args = parser.parse_args()

    # Get Atari games.
    env = gym.make('Pong-ram-v0')

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_ram_env(env, seed, args.cfg_name)

    main(env, int(4e7), args)
