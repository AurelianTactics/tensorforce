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
Pygame Learning Environment Integration: https://github.com/ntasfi/PyGame-Learning-Environment
"""

#to do:
#finish rest of args write up
#test pong: for lives/score issue (ie if 1 live, reset after every score conceded
#work my way though the functions on this list
    #verify that I fill out all the necessary environments that need to be filled out


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from ple import PLE
from ple.games.catcher import Catcher
from ple.games.pixelcopter import Pixelcopter
from ple.games.pong import Pong
from ple.games.puckworld import PuckWorld
from ple.games.raycastmaze import RaycastMaze
from ple.games.snake import Snake
from ple.games.waterworld import WaterWorld
from ple.games.monsterkong import MonsterKong
from ple.games.flappybird import FlappyBird


import pygame
import numpy as np
from tensorforce import TensorForceError
from tensorforce.environments import Environment


class PygameLearningEnvironment(Environment):

    def __init__(self, game_name, rewards, state_as_image = True, fps = 30, force_fps=True, frame_skip=2,
                 hold_action=2, visualize=False, width=84, height=84, lives=1):
        """
        Initialize Pygame Learning Environment
        https://github.com/ntasfi/PyGame-Learning-Environment

        Args:
            env_name: PLE environment

            fps: frames per second
            force_fps: False for slower speeds
            frame_skip: number of env frames to skip
            hold_action: number of env frames to hold each action for
            isRGB: get color or greyscale version of statespace #isRGB = False,
            game_height,game_width: height and width of environment
            visualize: If set True, the program will visualize the trainings, will slow down training
            lives: number of lives in game. Game resets on game over (ie lives = 0). only in Catcher and Pong (score)

        """

        self.env_name = game_name
        self.rewards = rewards
        self.lives = lives
        self.state_as_image = state_as_image
        self.fps = fps #30  # frames per second
        self.force_fps = force_fps #True  # False for slower speeds
        self.frame_skip = frame_skip  # frames to skip
        self.ple_num_steps = hold_action  # frames to continue action for
        #self.isRGB = isRGB #always returns color, lets tensorforce due the processing
        self.visualize = visualize
        self.width = width
        self.height = height
        #testing
        self.reached_terminal = 0
        self.episode_time_steps = 0
        self.episode_reward = 0
        self.total_time_steps = 0

        if self.env_name == 'catcher':
            self.game = Catcher(width=self.width, height=self.height,init_lives=self.lives)
        elif self.env_name == 'pixelcopter':
            self.game = Pixelcopter(width=self.width, height=self.height)
        elif self.env_name == 'pong':
            self.game = Pong(width=self.width, height=self.height,MAX_SCORE=self.lives)
        elif self.env_name == 'puckworld':
            self.game = PuckWorld(width=self.width, height=self.height)
        elif self.env_name == 'raycastmaze':
            self.game = RaycastMaze(width=self.width, height=self.height)
        elif self.env_name == 'snake':
            self.game = Snake(width=self.width, height=self.height)
        elif self.env_name == 'waterworld':
            self.game = WaterWorld(width=self.width, height=self.height)
        elif self.env_name == 'monsterkong':
            self.game = MonsterKong()
        elif self.env_name == 'flappybird':
            self.game = FlappyBird(width=144, height=256)  # limitations on height and width for flappy bird
        else:
            raise TensorForceError('Unknown Game Environement.')

        if self.state_as_image:
           process_state = None
        else:
            #create a preprocessor to read the state dictionary as a numpy array
            def process_state(state):
                # ret_value = np.fromiter(state.values(),dtype=float,count=len(state))
                ret_value = np.array(list(state.values()), dtype=np.float32)
                return ret_value

        # make a PLE instance
        self.env = PLE(self.game,reward_values=self.rewards,fps=self.fps, frame_skip=self.frame_skip,
                       num_steps=self.ple_num_steps,force_fps=self.force_fps,display_screen=self.visualize,
                       state_preprocessor = process_state)
        #self.env.init()
        #self.env.act(self.env.NOOP) #game starts on black screen
        #self.env.reset_game()
        #self.env.act(self.env.NOOP)
        #self.env.act(self.env.NOOP)
        #self.env.act(self.env.NOOP)
        #self.env.act(self.env.NOOP)
        #self.env.reset_game()


        # setup gamescreen object
        if state_as_image:
            w, h = self.env.getScreenDims()
            self.gamescreen = np.empty((h, w, 3), dtype=np.uint8)
        else:
            self.gamescreen = np.empty(self.env.getGameStateDims(), dtype=np.float32)
        # if isRGB:
        #     self.gamescreen = np.empty((h, w, 3), dtype=np.uint8)
        # else:
        #     self.gamescreen = np.empty((h, w), dtype=np.uint8)

        # setup action converter
        # PLE returns legal action indexes, convert these to just numbers
        self.action_list = self.env.getActionSet()
        self.action_list = sorted(self.action_list, key=lambda x: (x is None, x))



    def __str__(self):
        return 'PygameLearningEnvironment({})'.format(self.env_name)

    def close(self):
        pygame.quit()
        self.env = None

    def reset(self):
        # if isinstance(self.gym, gym.wrappers.Monitor):
        #     self.gym.stats_recorder.done = True
        #env.act(env.NOOP) # need to take an action or screen is black
        # clear gamescreen
        if self.state_as_image:
            self.gamescreen = np.empty(self.gamescreen.shape, dtype=np.uint8)
        else:
            self.gamescreen = np.empty(self.gamescreen.shape, dtype=np.float32)
        self.env.reset_game()
        return self.current_state

    def execute(self, actions):

        #print("lives check in ple {}".format(self.env.lives()))
        #self.env.saveScreen("test_screen_capture_before_{}.png".format(self.total_time_steps))
        #lives_check = self.env.lives() #testing code

        ple_actions = self.action_list[actions]
        reward = self.env.act(ple_actions)
        state = self.current_state
        # testing code
        # self.env.saveScreen("test_screen_capture_after_{}.png".format(self.total_time_steps))
        # self.episode_time_steps += 1
        # self.episode_reward += reward
        # self.total_time_steps += 1
        # print("reward is {}".format(reward))
        # #if self.env.lives() != lives_check:
        # #    print('lives are different is game over? {}'.format(self.env.game_over()))
        # print('lives {}, game over {}, old lives {}'.format(self.env.lives(),self.env.game_over(),lives_check))

        if self.env.game_over():
            terminal = True
            # testing code
            self.reached_terminal += 1
            # print("GAME OVER reached terminal {}".format(self.reached_terminal))
            # print("episode time steps {}, episode reward {}".format(self.episode_time_steps,self.episode_reward))
            # self.episode_reward = 0
            # self.episode_time_steps = 0
            # print("total timesteps {}".format(self.total_time_steps))
        else:
            terminal = False

        return state, terminal, reward

    @property
    def actions(self):
        return dict(type='int', num_actions=len(self.action_list), names=self.action_list)

    # @property
    # def actions(self):
    #     return OpenAIGym.action_from_space(space=self.gym.action_space)

    #ALE implementation
    # @property
    # def actions(self):
    #     return dict(type='int', num_actions=len(self.action_inds), names=self.action_names)

    @property
    def states(self):
        return dict(shape=self.gamescreen.shape, type=float)

    @property
    def current_state(self):
        #returned state can either be an image or an np array of key components
        if self.state_as_image:
            self.gamescreen = self.env.getScreenRGB()
            # if isRGB:
            #     self.gamescreen = self.env.getScreenRGB()
            # else:
            #     self.gamescreen = self.env.getScreenGrayscale()
        else:
            self.gamescreen = self.env.getGameState()

        return np.copy(self.gamescreen)

    #ALE implementation
    # @property
    # def states(self):
    #     return dict(shape=self.gamescreen.shape, type=float)

    # @property
    # def current_state(self):
    #     self.gamescreen = self.ale.getScreenRGB(self.gamescreen)
    #     return np.copy(self.gamescreen)

    # @property
    # def is_terminal(self):
    #     if self.loss_of_life_termination and self.life_lost:
    #         return True
    #     else:
    #         return self.ale.game_over()