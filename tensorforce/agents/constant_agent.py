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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.models.constant_model import ConstantModel


class ConstantAgent(Agent):
    """
    Agent returning constant action values.
    """

    def __init__(
        self,
        states,
        actions,
        action_values,
        batched_observe=True,
        batching_capacity=1000,
        scope='constant',
        device=None,
        saver=None,
        summarizer=None,
        distributed=None
    ):
        """
        Initializes the constant agent.

        Args:
            action_values (value, or dict of values): Action values returned by the agent
                (required).
            scope (str): TensorFlow scope (default: name of agent).
            device: TensorFlow device (default: none)
            saver (spec): Saver specification, with the following attributes (default: none):
                - directory: model directory.
                - file: model filename (optional).
                - seconds or steps: save frequency (default: 600 seconds).
                - load: specifies whether model is loaded, if existent (default: true).
                - basename: optional file basename (default: 'model.ckpt').
            summarizer (spec): Summarizer specification, with the following attributes (default:
                none):
                - directory: summaries directory.
                - seconds or steps: summarize frequency (default: 120 seconds).
                - labels: list of summary labels to record (default: []).
                - meta_param_recorder_class: ???.
            distributed (spec): Distributed specification, with the following attributes (default:
                none):
                - cluster_spec: TensorFlow ClusterSpec object (required).
                - task_index: integer (required).
                - parameter_server: specifies whether this instance is a parameter server (default:
                    false).
                - protocol: communication protocol (default: none, i.e. 'grpc').
                - config: TensorFlow ConfigProto object (default: none).
                - replica_model: internal.
        """

        self.scope = scope
        self.device = device
        self.saver = saver
        self.summarizer = summarizer
        self.distributed = distributed
        self.batching_capacity = batching_capacity
        self.action_values = action_values

        super(ConstantAgent, self).__init__(
            states=states,
            actions=actions,
            batched_observe=batched_observe,
            batching_capacity=batching_capacity
        )

    def initialize_model(self):
        return ConstantModel(
            states=self.states,
            actions=self.actions,
            scope=self.scope,
            device=self.device,
            saver=self.saver,
            summarizer=self.summarizer,
            distributed=self.distributed,
            batching_capacity=self.batching_capacity,
            action_values=self.action_values
        )
