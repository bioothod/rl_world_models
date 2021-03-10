from typing import *

import gym
import logging
import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from tensorflow.keras.mixed_precision import experimental as mixed_precision

logger = logging.getLogger('sac')

class ValueNetwork(tf.keras.Model):
    def __init__(self, hidden_size: int, **kwargs: dict):
        super().__init__(**kwargs)

        self.dense0 = tf.keras.layers.Dense(hidden_size, name=f'{self.name}/dense0')
        self.dense1 = tf.keras.layers.Dense(hidden_size, name=f'{self.name}/dense1')
        self.dense2 = tf.keras.layers.Dense(1, name=f'{self.name}/dense2')

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.dense0(inputs)
        x = tf.nn.relu(x)

        x = self.dense1(x)
        x = tf.nn.relu(x)

        x = self.dense2(x)
        return x

class SoftQNetwork(tf.keras.Model):
    def __init__(self, hidden_size: int, **kwargs: dict):
        super().__init__(**kwargs)

        self.dense0 = tf.keras.layers.Dense(hidden_size, name=f'{self.name}/dense0')
        self.dense1 = tf.keras.layers.Dense(hidden_size, name=f'{self.name}/dense1')
        self.dense2 = tf.keras.layers.Dense(1, name=f'{self.name}/dense2')

    def __call__(self, states: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        x = tf.concat([states, actions], 1)

        x = self.dense0(x)
        x = tf.nn.relu(x)

        x = self.dense1(x)
        x = tf.nn.relu(x)

        x = self.dense2(x)
        return x

class PolicyNetwork(tf.keras.Model):
    def __init__(self,
            hidden_size: int,
            num_actions: int,
            tfp_seed: tfp.util.SeedStream,
            log_std_min: float = -20,
            log_std_max: float = 2,
            **kwargs: dict):
        super().__init__(**kwargs)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.tfp_seed = tfp_seed

        self.dense0 = tf.keras.layers.Dense(hidden_size, name=f'{self.name}/dense0')
        self.dense1 = tf.keras.layers.Dense(hidden_size, name=f'{self.name}/dense1')

        self.mean_linear = tf.keras.layers.Dense(num_actions, name=f'{self.name}/mean_linear')
        self.log_std_linear = tf.keras.layers.Dense(num_actions, name=f'{self.name}/log_std_linear')

    def __call__(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.dense0(inputs)
        x = tf.nn.relu(x)

        x = self.dense1(x)
        x = tf.nn.relu(x)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, states: tf.Tensor, epsilon: float = 1e-8) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        mean, log_std = self(states)
        std = tf.math.exp(log_std)

        dist = tfp.distributions.Normal(loc=mean, scale=std)
        action_unscaled = dist.sample(seed=self.tfp_seed)
        action = tf.math.tanh(action_unscaled)

        log_prob = dist.log_prob(action_unscaled) - tf.reduce_sum(tf.math.log(1 - tf.math.pow(action, 2) + epsilon), axis=1, keepdims=True)

        return action, log_prob

    def get_action(self, state: tf.Tensor) -> tf.Tensor:
        states = tf.expand_dims(state, 0)

        mean, log_std = self(states)
        std = tf.math.exp(log_std)

        dist = tfp.distributions.Normal(loc=mean, scale=std)
        action_unscaled = dist.sample(seed=self.tfp_seed)
        action = tf.math.tanh(action_unscaled)

        return action[0]

class ReplayBuffer:
    def __init__(self, max_size: int, dtype: tf.dtypes.DType):
        self.max_size = max_size
        self.data_type = dtype

        self.replay_buffer = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
        self.write_index = 0

    def push(self, data: tf.Tensor):
        self.replay_buffer = self.replay_buffer.write(self.write_index, data)
        self.write_index += 1
        if self.write_index == self.max_size:
            self.write_index = 0

    def size(self):
        return self.replay_buffer.size()

    def all(self):
        return self.replay_buffer.stack()

    # tf.TensorArray() can be be read or sampled in graph mode if it hasn't been created in graph mode
    def sample(self, sample_index):
        buf = self.replay_buffer.stack()
        return tf.gather(buf, sample_index)

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
        self.reward.push(reward)
        self.new_state.push(new_state)
        self.done.push(done)

    def size(self):
        return self.state.size()

    @tf.function
    def sample(self, size):
        sample_index = self.sample_index[:self.size()]
        sample_index = tf.random.shuffle(sample_index)
        sample_index = sample_index[:size]

        state = self.state.sample(sample_index)
        action = self.action.sample(sample_index)
        reward = self.reward.sample(sample_index)
        new_state = self.new_state.sample(sample_index)
        done = self.done.sample(sample_index)

        return state, action, reward, new_state, done

class SoftActorCritic:
    def __init__(self,
            env,
            hidden_size: int,
            batch_size: int,
            initial_learning_rate: float = 1e-2,
            min_learning_rate: float = 1e-5,

            dtype: tf.dtypes.DType = tf.float32,
            gamma: float = 0.99,
            soft_tau: float = 1e-2,
            replay_buffer_size: int = 1_000_000,
            max_steps_per_episode: int = 1000,
            tfp_seed: tfp.util.SeedStream = tfp.util.SeedStream(None, ''),
            ):

        self.env = env
        self.batch_size = batch_size

        self.gamma = gamma
        self.soft_tau = soft_tau
        self.max_steps_per_episode = max_steps_per_episode
        self.data_type = dtype

        self.min_learning_rate = min_learning_rate

        num_actions = env.action_space.shape[0]

        self.replay_buffer = StateReplayBuffer(max_size=replay_buffer_size, dtype=dtype)

        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.learning_rate = tf.Variable(initial_learning_rate, dtype=tf.float32, name='learning_rate')
        self.epoch_var = tf.Variable(0, dtype=tf.int64, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        self.value_net = ValueNetwork(hidden_size, name='value_net')
        self.target_value_net = ValueNetwork(hidden_size, name='target_value_net')
        self.update_weights(self.target_value_net, self.value_net, 1.0)

        self.soft_q_net0 = SoftQNetwork(hidden_size, name='soft_q0_net')
        self.soft_q_net1 = SoftQNetwork(hidden_size, name='soft_q1_net')

        self.policy_net = PolicyNetwork(hidden_size, num_actions, tfp_seed, name='policy_net')

        self.value_loss = tf.keras.losses.MeanSquaredError()
        self.soft_q_loss = tf.keras.losses.MeanSquaredError()
        #self.value_loss = tf.keras.losses.Huber()
        #self.soft_q_loss = tf.keras.losses.Huber()

        self.value_metric = tf.keras.metrics.MeanSquaredError()
        self.soft_q0_metric = tf.keras.metrics.MeanSquaredError()
        self.soft_q1_metric = tf.keras.metrics.MeanSquaredError()
        self.policy_metric = tf.keras.metrics.Mean()

        self.value_opt = self.create_optimizer()
        self.soft_q_opt0 = self.create_optimizer()
        self.soft_q_opt1 = self.create_optimizer()
        self.policy_opt = self.create_optimizer()


    def create_optimizer(self):
        if False:
            opt = tfa.optimizers.RectifiedAdam(lr=self.learning_rate, min_lr=self.min_learning_rate)
            opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if self.data_type == tf.float16:
            opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

        return opt


    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, reward, done, _ = self.env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.int32, tf.int32])

    def tf_env_sample_action(self) -> tf.Tensor:
        return tf.numpy_function(lambda: self.env.action_space.sample(), [], [tf.float32])

    @tf.function
    def update_weights(self, target, model, soft_tau):
        for target_var, var in zip(target.trainable_variables, model.trainable_variables):
            v = target_var * (1.0 - soft_tau) + var * soft_tau
            target_var.assign(v)

    @tf.function
    def train_step(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        with tf.GradientTape(persistent=True) as tape:
            pred_q0 = self.soft_q_net0(state, action)
            pred_q1 = self.soft_q_net1(state, action)
            new_action, log_prob = self.policy_net.evaluate(state)

            target_value = self.target_value_net(next_state)
            target_q_value = tf.stop_gradient(reward + (1 - done) * self.gamma * target_value)

            q_loss0 = self.soft_q_loss(pred_q0, target_q_value)
            q_loss1 = self.soft_q_loss(pred_q1, target_q_value)

            pred_new_q0 = self.soft_q_net0(state, new_action)
            pred_new_q1 = self.soft_q_net1(state, new_action)
            pred_new_q_value = tf.minimum(pred_new_q0, pred_new_q1)
            target_value_func = pred_new_q_value - log_prob

            pred_value = self.value_net(state)
            value_loss = self.value_loss(target_value_func, pred_value)

            policy_loss = tf.math.reduce_mean(log_prob - pred_new_q_value)

            with tape.stop_recording():
                grads = tape.gradient(value_loss, self.value_net.trainable_variables)
                self.value_opt.apply_gradients(zip(grads, self.value_net.trainable_variables))

                self.value_metric.update_state(pred_value, target_value_func)

            with tape.stop_recording():
                grads = tape.gradient(q_loss0, self.soft_q_net0.trainable_variables)
                self.soft_q_opt0.apply_gradients(zip(grads, self.soft_q_net0.trainable_variables))
                grads = tape.gradient(q_loss1, self.soft_q_net1.trainable_variables)
                self.soft_q_opt1.apply_gradients(zip(grads, self.soft_q_net1.trainable_variables))

                self.soft_q0_metric.update_state(pred_q0, target_q_value)
                self.soft_q1_metric.update_state(pred_q1, target_q_value)


            with tape.stop_recording():
                grads = tape.gradient(policy_loss, self.policy_net.trainable_variables)
                self.policy_opt.apply_gradients(zip(grads, self.policy_net.trainable_variables))

                self.policy_metric.update_state(policy_loss)

        del tape

        self.update_weights(self.target_value_net, self.value_net, self.soft_tau)


    def run_episode(self, initial_state: tf.Tensor):
        state = initial_state
        avg_reward = tf.constant(0, dtype=self.data_type)
        episode_reward = tf.constant(0, dtype=self.data_type)

        self.policy_metric.reset_states()
        self.value_metric.reset_states()
        self.soft_q0_metric.reset_states()
        self.soft_q1_metric.reset_states()


        for t in tf.range(self.max_steps_per_episode):
            if self.global_step > 1000:
                action = self.policy_net.get_action(state)
            else:
                action = self.tf_env_sample_action()

            next_state, reward, done = self.tf_env_step(action)
            state = tf.cast(state, self.data_type)
            action = tf.cast(action, self.data_type)
            next_state = tf.cast(next_state, self.data_type)
            reward = tf.cast(reward, self.data_type)
            done = tf.cast(done, self.data_type)

            episode_reward += reward

            #tf.print('state:', tf.shape(state), ', action:', tf.shape(action), ', reward:', tf.shape(reward), ', next_state:', tf.shape(next_state), ', done:', tf.shape(done))
            self.replay_buffer.push(state, action, reward, next_state, done)

            if self.global_step == 0:
                self.update_weights(self.target_value_net, self.value_net, 1)

            if self.replay_buffer.size() >= self.batch_size:
                self.train_step()

            self.global_step.assign_add(1)
            state = next_state

            if tf.cast(done, tf.bool):
                break

        self.epoch_var.assign_add(1)
        return episode_reward


def main(env, tfp_seed):
    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)


    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    use_fp16 = False
    initial_learning_rate = 3e-4
    min_learning_rate = 1e-5

    dtype = tf.float32
    np_dtype = np.float32
    if use_fp16:
        dtype = tf.float16
        np_dtype = np.float16

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    #checkpoint = tf.train.Checkpoint(step=global_step, epoch=epoch_var, optimizer=opt, model=model)
    #manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=20)

    max_steps_per_episode = 1000

    hidden_size = 32
    batch_size = 128

    replay_buffer_size = 300_000
    soft_tau = 1e-2

    sac = SoftActorCritic(
            env = env,
            hidden_size = hidden_size,
            batch_size = batch_size,
            initial_learning_rate = initial_learning_rate,
            min_learning_rate = min_learning_rate,
            dtype = dtype,
            soft_tau = soft_tau,
            replay_buffer_size = replay_buffer_size,
            max_steps_per_episode = max_steps_per_episode,
            tfp_seed = tfp_seed,
    )

    rewards = []
    for episode in range(1000):
        initial_state = env.reset()
        initial_state = initial_state.astype(np_dtype)

        episode_reward = sac.run_episode(initial_state)

        episode_reward = episode_reward.numpy().astype(int)

        rewards.append(episode_reward)
        if len(rewards) > 200:
            rewards = rewards[1:]
        avg_reward = np.mean(rewards)

        logger.info(f'{episode:4d}: step: {sac.global_step.numpy():5d}, episode_reward: {episode_reward:5d}, avg_reward: {avg_reward:5.2f}, losses: '
                f'policy: {sac.policy_metric.result():3.2f}, '
                f'value: {sac.value_metric.result():2.2f}, '
                f'soft_q0: {sac.soft_q0_metric.result():2.2f}, '
                f'soft_q1: {sac.soft_q1_metric.result():2.2f}')

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return actions

if __name__ == "__main__":
    env = NormalizedActions(gym.make("Pendulum-v0"))
    #env = gym.make("Pendulum-v0")

    seed = 42
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tfp_seed = tfp.util.SeedStream(seed, '42')
    random.seed(seed)

    main(env, tfp_seed)
