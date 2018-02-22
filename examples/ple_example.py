# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Arcade Learning Environment execution
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import time
import numpy as np

from tensorforce import TensorForceError
import json

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.pygame_learning_environment import PygameLearningEnvironment
from tensorforce.core.preprocessors import Preprocessor


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('game_name', help="Game name for environment to load")

    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default=None, help="Network specification file")
    parser.add_argument('-e', '--episodes', type=int, default=50000, help="Number of episodes")
    parser.add_argument('-t', '--max_timesteps', type=int, default=2000, help="Maximum number of timesteps per episode")
    parser.add_argument('-s', '--save', help="Save agent to this dir")
    parser.add_argument('-se', '--save_episodes', type=int, default=10000, help="Save agent every x episodes")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

    parser.add_argument('--state_as_image', type=int, default=1, help="get game state as screenshots or component locations")
    parser.add_argument('--fps', type=int,default=30, help="frames per second")
    parser.add_argument('--force_fps', type=int, default=1, help="How FPS interacts with game's clock. Runs unevenly and slower on False.")
    parser.add_argument('--frame_skip', type=int, default=2, help="skip this many frames between states")
    parser.add_argument('--hold_action', type=int, default=2, help="continue action for this many frames")
    parser.add_argument('--visualize', type=int, default=0, help="display emulator screen")
    parser.add_argument('--width', type=int, default=84, help="screen width")
    parser.add_argument('--height', type=int, default=84, help="screen height")
    parser.add_argument('--rew_pos',type=float, default=float(0.1),help="positive reward")
    parser.add_argument('--rew_neg', type=float, default=float(-0.1), help="negative reward")
    parser.add_argument('--rew_tick', type=float, default=float(-0.0), help="reward for timestep")
    parser.add_argument('--rew_win', type=float, default=float(1.0), help="reward for game win")
    parser.add_argument('--rew_loss', type=float, default=float(-1.0), help="reward for game loss")

    args = parser.parse_args()

    #adjust rewards each agent receives
    rewards = {
        "tick": float(args.rew_tick),  # each time the game steps forward
        "positive": float(args.rew_pos),  # reward for positive event
        "negative": float(args.rew_neg),  # reward for negative event
        "loss" : float(args.rew_loss), #reward for loss of game
        "win" : float(args.rew_win) #reward for win game
    }

    #preprocessing config
    if args.state_as_image:
        preprocessing_config = [
            {
                "type": "grayscale"
            }, {
                "type": "divide",
                "scale": 255.0
            }, {
                "type": "sequence",
                "length": 4
            }
        ]
    else:
        #batch of state variables, standardize
        preprocessing_config = [
            {
                "type": "running_standardize",
                "reset_after_batch": False
            }
        ]

    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)  # configurable!!!
    # logger.addHandler(logging.StreamHandler(sys.stdout))

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename="ple_test.log", level=logging.INFO)

    environment = PygameLearningEnvironment(args.game_name,rewards=rewards,state_as_image=bool(args.state_as_image),
                                            fps=args.fps, force_fps=bool(args.force_fps),frame_skip=args.frame_skip,
                                            hold_action=args.hold_action,visualize=bool(args.visualize),
                                            width=args.width,height=args.height)

    #print(environment.states)
    #print(environment.actions)

    if args.agent_config is not None:
        with open(args.agent_config, 'r') as fp:
            agent_config = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network_spec is not None:
        with open(args.network_spec, 'r') as fp:
            network_spec = json.load(fp=fp)
    else:
        network_spec = None
        logger.info("No network configuration provided.")

    if args.save:
        save_dir = os.path.dirname(args.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

        ts = int(time.time())
        save_dict = {
            'directory': "{}".format(args.save),
            'file': "{}_{}".format(args.game_name,ts),
            'steps': args.save_episodes,
            'load': False,
            #"basename": "base"
        }
        #agent.saver = save_dict
    else:
        save_dict = None

    # summarizer_config = {
    #     "directory": "/tmp/ple_tb_example",
    #     "labels": ["episode_rewards", "learning_rate", "epsilon"],
    #     "steps": 50
    # }

    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states=environment.states,
            actions=environment.actions,
            network=network_spec,
            saver=save_dict,
            #summarizer=summarizer_config,
            states_preprocessing=preprocessing_config

        )
    )

    if args.debug:
        logger.info("-" * 16)
        logger.info("Configuration:")
        logger.info(agent_config)

    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)



    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    attrs = vars(agent)
    for item in attrs.items():
        print(item)


    report_episodes = args.episodes // 1000
    if report_episodes < 1:
        report_episodes = 1
    if args.debug:
        report_episodes = 1

    def episode_finished(r):
        # attrs = vars(r)
        # for item in attrs.items():
        #     print(item)
        #print(len(r.episode_rewards))
        #print(np.shape(r.episode_rewards))
        #print(r.episode_rewards)
        if r.episode % report_episodes == 0:
            sps = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}".format(ep=r.episode, ts=r.timestep, sps=sps))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
            #logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
            logger.info("Avg of last 500 timesteps: {}".format(sum(r.episode_timesteps[-500:]) / 500))
        return True

    logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))
    runner.run(timesteps=None, episodes=args.episodes,
               max_episode_timesteps=args.max_timesteps, episode_finished=episode_finished)

    if args.save:
        save_dir = os.path.dirname(args.save)
        ts = int(time.time())
        agent.save_model('{}/final_{}_{}'.format(save_dir,args.game_name, ts))

    runner.close()
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

    environment.close()


if __name__ == '__main__':
    main()
