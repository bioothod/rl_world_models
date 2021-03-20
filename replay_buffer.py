#from __future__ import annotations

from typing import *

import tensorflow as tf

class ReplayBuffer:
    def __init__(self, max_size: int, dtype: tf.dtypes.DType) -> None:
        self.max_size = max_size
        self.data_type = dtype

        self.replay_buffer = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
        self.write_index = 0

    def push(self, data: tf.Tensor) -> None:
        self.replay_buffer = self.replay_buffer.write(self.write_index, data)
        self.write_index += 1
        if self.write_index == self.max_size:
            self.write_index = 0

    def size(self) -> int:
        return self.replay_buffer.size()

    def all(self) -> tf.Tensor:
        return self.replay_buffer.stack()

    def sample(self, sample_index):
        return self.replay_buffer.gather(sample_index)

class StateReplayBuffer:
    def __init__(self, max_size: int, dtype: tf.dtypes.DType):
        self.state = ReplayBuffer(max_size, dtype)
        self.action = ReplayBuffer(max_size, dtype)
        self.reward = ReplayBuffer(max_size, dtype)
        self.new_state = ReplayBuffer(max_size, dtype)
        self.done = ReplayBuffer(max_size, dtype)

        self.sample_index = tf.range(max_size)

    def push(self, state: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, new_state: tf.Tensor, done: tf.Tensor) -> None:
        self.state.push(state)
        self.action.push(action)
        self.reward.push(tf.reshape(reward, [1]))
        self.new_state.push(new_state)
        self.done.push(tf.reshape(done, [1]))

    def size(self) -> int:
        return self.state.size()

    def sample(self, size) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        sample_index = self.sample_index[:self.size()]
        sample_index = tf.random.shuffle(sample_index)
        sample_index = sample_index[:size]

        state = self.state.sample(sample_index)
        action = self.action.sample(sample_index)
        reward = self.reward.sample(sample_index)
        new_state = self.new_state.sample(sample_index)
        done = self.done.sample(sample_index)

        return state, action, reward, new_state, done
