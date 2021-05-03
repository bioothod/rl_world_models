#from __future__ import annotations

from numbers import Number
from typing import *

import gym
import logging
import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
tfd = tfp.distributions

from utils.gym import is_continuous_space, is_discrete_space
from replay_buffer import StateReplayBuffer, ReplayBuffer

logger = logging.getLogger('sac')

class Critic(tf.keras.Model):
    def __init__(self, hidden_size: int, **kwargs):
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

EPSILON = 1e-8

def gaussian_likelihood(input_, mu_, log_std):
    """
    Helper to computer log likelihood of a gaussian.
    Here we assume this is a Diagonal Gaussian.

    :param input_: (tf.Tensor)
    :param mu_: (tf.Tensor)
    :param log_std: (tf.Tensor)
    :return: (tf.Tensor)
    """
    return -0.5 * (((input_ - mu_) / (tf.exp(log_std) + EPSILON)) ** 2 + 2 * log_std + np.log(2 * np.pi))


def gaussian_entropy(log_std):
    """
    Compute the entropy for a diagonal gaussian distribution.

    :param log_std: (tf.Tensor) Log of the standard deviation
    :return: (tf.Tensor)
    """
    return tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)


def clip_but_pass_gradient(input_, lower=-1., upper=1.):
    clip_up = tf.cast(input_ > upper, tf.float32)
    clip_low = tf.cast(input_ < lower, tf.float32)
    return input_ + tf.stop_gradient((upper - input_) * clip_up + (lower - input_) * clip_low)

def apply_squashing_func(mu_, pi_, logp_pi):
    """
    Squash the output of the Gaussian distribution
    and account for that in the log probability
    The squashed mean is also returned for using
    deterministic actions.

    :param mu_: (tf.Tensor) Mean of the gaussian
    :param pi_: (tf.Tensor) Output of the policy before squashing
    :param logp_pi: (tf.Tensor) Log probability before squashing
    :return: ([tf.Tensor])
    """
    # Squash the output
    deterministic_policy = tf.tanh(mu_)
    policy = tf.tanh(pi_)
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    # logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    logp_pi -= tf.math.reduce_sum(tf.math.log(1 - tf.math.pow(policy, 2) + EPSILON), axis=1, keepdims=True)
    return deterministic_policy, policy, logp_pi


class Actor(tf.keras.Model):
    def __init__(self,
            hidden_size: int,
            num_actions: int,
            log_std_min: float = -1e10,
            log_std_max: float = 1,
            **kwargs):
        super().__init__(**kwargs)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.dense0 = tf.keras.layers.Dense(hidden_size, name=f'{self.name}/dense0')
        self.dense1 = tf.keras.layers.Dense(hidden_size, name=f'{self.name}/dense1')

        self.mean_linear = tf.keras.layers.Dense(num_actions, activation='tanh', name=f'{self.name}/mean_linear')
        self.log_std_linear = tf.keras.layers.Dense(num_actions, name=f'{self.name}/log_std_linear')

    def __call__(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.dense0(inputs)
        x = tf.nn.relu(x)

        x = self.dense1(x)
        x = tf.math.tanh(x)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action_log_prob(self, states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mean, log_std = self(states)
        std = tf.math.exp(log_std)

        #dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
        dist = tfd.Normal(loc=mean, scale=std)

        action_unscaled = dist.sample()
        #action_unscaled = tf.clip_by_value(action_unscaled, -1e10, 0.5)

        log_prob_sample = gaussian_likelihood(action_unscaled, mean, log_std)

        action = tf.math.tanh(action_unscaled)

        log_prob = log_prob_sample - tf.reduce_sum(tf.math.log(clip_but_pass_gradient(1 - action ** 2, lower=0, upper=1) + EPSILON), axis=1, keepdims=True)

        #tf.print('dist max: states: ', tf.reduce_max(states), 'mean: ', tf.reduce_max(mean), ', log_std:', tf.reduce_max(log_std), 'action_unscaled:', tf.reduce_max(action_unscaled), ', action:', tf.reduce_max(action), ', log_prob:', tf.reduce_max(log_prob), ', log_prob_sample:', tf.reduce_max(log_prob_sample))

        #log_prob = log_prob_sample - tf.math.log(1 - tf.math.pow(action, 2) + EPSILON)
        #log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)

        return action, log_prob

    def get_action(self, state: tf.Tensor) -> tf.Tensor:
        states = tf.expand_dims(state, 0)
        action, _ = self.action_log_prob(states)

        #action = tf.squeeze(action, 1)
        action = tf.squeeze(action, 0)
        return action

def heuristic_target_entropy(action_space):
    if is_continuous_space(action_space):
        heuristic_target_entropy = -np.prod(action_space.shape)
    elif is_discrete_space(action_space):
        raise NotImplementedError('Discrete action space is not supported')
    else:
        raise NotImplementedError((type(action_space), action_space))

    return heuristic_target_entropy

class SoftActorCritic:
    def __init__(self,
            env,
            hidden_size: int,
            batch_size: int,
            initial_learning_rate: float = 1e-2,
            min_learning_rate: float = 1e-5,

            dtype: tf.dtypes.DType = tf.float32,
            reward_scale: float = 1.0,
            gamma: float = 0.99,
            soft_tau: float = 1e-2,
            gradient_norm_clip: float = 5,
            replay_buffer_size: int = 1_000_000,
            max_steps_per_episode: int = 1000,
            warmup_sample_steps: int = 1000,
            steps_per_training_step: int = 1,
            ):

        self.env = env
        self.batch_size = batch_size

        self.gamma = gamma
        self.soft_tau = soft_tau
        self.reward_scale = reward_scale

        self.max_steps_per_episode = max_steps_per_episode
        self.data_type = dtype
        self.warmup_sample_steps = tf.constant(warmup_sample_steps, dtype=tf.int64)
        self.steps_per_training_step = steps_per_training_step
        self.gradient_norm_clip = gradient_norm_clip

        self.min_learning_rate = min_learning_rate

        num_actions = env.action_space.shape[0]

        self.replay_buffer = StateReplayBuffer(max_size=replay_buffer_size, dtype=dtype)

        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.learning_rate = tf.Variable(initial_learning_rate, dtype=tf.float32, name='learning_rate')
        self.epoch_var = tf.Variable(0, dtype=tf.int64, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        num_qs = 2
        self.qs = []
        self.tqs = []
        self.q_opts = []
        self.q_metrics = []
        for idx in range(num_qs):
            q = Critic(hidden_size, name=f'q{idx}')
            tq = Critic(hidden_size, name=f'target_q{idx}')

            self.update_weights(tq, q, 1)

            self.qs.append(q)
            self.tqs.append(tq)

            self.q_metrics.append(tf.keras.metrics.Mean())
            self.q_opts.append(self.create_optimizer())


        self.policy = Actor(hidden_size, num_actions, name='policy')
        self.policy_metric = tf.keras.metrics.Mean()
        self.policy_opt = self.create_optimizer()

        self.log_alpha = tf.Variable(0.0, dtype=dtype)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)
        self.alpha_metric = tf.keras.metrics.Mean()
        self.alpha_opt = self.create_optimizer()

        self.target_entropy = heuristic_target_entropy(env.action_space)

    def create_optimizer(self):
        if False:
            opt = tfa.optimizers.RectifiedAdam(lr=self.learning_rate, min_lr=self.min_learning_rate)
            opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if self.data_type == tf.float16:
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

        return opt


    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, reward, done, _ = self.env.step(action)
        return (state.astype(np.float32),
                np.array(reward, np.float32),
                np.array(done, np.float32))

    def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action], [tf.float32, tf.float32, tf.float32])

    def tf_env_sample_action(self) -> tf.Tensor:
        return tf.numpy_function(lambda: self.env.action_space.sample(), [], [tf.float32])

    @tf.function
    def update_weights(self, target, model, soft_tau):
        for target_var, var in zip(target.trainable_variables, model.trainable_variables):
            v = target_var * (1.0 - soft_tau) + var * soft_tau
            target_var.assign(v)

    def compute_q_targets(self, states, actions, rewards, next_states, dones):
        next_actions, next_log_probs = self.policy.action_log_prob(next_states)
        next_qs = [q(next_states, next_actions) for q in self.tqs]
        next_q_values = tf.reduce_min(next_qs, axis=0)

        entropy_esitmate = tf.convert_to_tensor(self.alpha)
        next_values = next_q_values - entropy_esitmate * next_log_probs
        q_targets = rewards * self.reward_scale + self.gamma * (1 - dones) * next_values
        return tf.stop_gradient(q_targets)

    @tf.function(experimental_relax_shapes=True)
    def update_critic(self, states, actions, rewards, next_states, dones):
        q_targets = self.compute_q_targets(states, actions, rewards, next_states, dones)

        ret_q_values = []
        ret_q_losses = []

        for q, opt, metric in zip(self.qs, self.q_opts, self.q_metrics):
            with tf.GradientTape() as tape:
                q_values = q(states, actions)
                q_losses = 0.5 * tf.keras.losses.MSE(q_targets, q_values)
                q_loss = tf.nn.compute_average_loss(q_losses)

            grads = tape.gradient(q_loss, q.trainable_variables)
            #grads, _ = tf.clip_by_global_norm(grads, self.gradient_norm_clip)
            opt.apply_gradients(zip(grads, q.trainable_variables))

            metric.update_state(q_loss)
            ret_q_values.append(q_values)
            ret_q_losses.append(q_losses)

        return ret_q_values, ret_q_losses

    @tf.function(experimental_relax_shapes=True)
    def update_actor(self, states):
        entropy_esitmate = tf.convert_to_tensor(self.alpha)

        with tf.GradientTape() as tape:
            next_actions, next_log_probs = self.policy.action_log_prob(states)
            q_log_targets = [q(states, next_actions) for q in self.qs]
            q_log_target = tf.reduce_min(q_log_targets, axis=0)

            policy_losses = entropy_esitmate * next_log_probs - q_log_target
            policy_loss = tf.nn.compute_average_loss(policy_losses)

        grads = tape.gradient(policy_loss, self.policy.trainable_variables)
        #grads, _ = tf.clip_by_global_norm(grads, self.gradient_norm_clip)
        self.policy_opt.apply_gradients(zip(grads, self.policy.trainable_variables))

        self.policy_metric.update_state(policy_loss)

        return policy_losses

    @tf.function
    def update_alpha(self, states):
        #if not isinstance(self.target_entropy, Number):
        #    return 0.0

        next_actions, next_log_probs = self.policy.action_log_prob(states)

        with tf.GradientTape() as tape:
            alpha_losses = -1 * self.alpha * tf.stop_gradient(next_log_probs + self.target_entropy)
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        variables = [self.log_alpha]
        grads = tape.gradient(alpha_loss, variables)
        #grads, _ = tf.clip_by_global_norm(grads, self.gradient_norm_clip)
        self.alpha_opt.apply_gradients(zip(grads, variables))

        self.alpha_metric.update_state(alpha_loss)

        return alpha_losses

    def train_step(self) -> None:
        for steps in range(self.steps_per_training_step):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            q_values, q_losses = self.update_critic(states, actions, rewards, next_states, dones)
            actor_loss = self.update_actor(states)
            alpha_loss = self.update_alpha(states)

            for q, tq in zip(self.qs, self.tqs):
                self.update_weights(tq, q, self.soft_tau)

    def run_episode(self, initial_state: tf.Tensor) -> tf.Tensor:
        state = initial_state
        avg_reward = tf.constant(0, dtype=self.data_type)
        episode_reward = tf.constant(0, dtype=self.data_type)

        self.policy_metric.reset_states()
        self.alpha_metric.reset_states()
        [q.reset_states() for q in self.q_metrics]


        for t in tf.range(self.max_steps_per_episode):
            if self.global_step > self.warmup_sample_steps:
                action = self.policy.get_action(state)
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

            if self.replay_buffer.size() >= self.batch_size and self.global_step > self.warmup_sample_steps:
                self.train_step()

            self.global_step.assign_add(1)
            state = next_state

            if tf.cast(done, tf.bool):
                break

        self.epoch_var.assign_add(1)
        return episode_reward


def main(env):
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

        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    #checkpoint = tf.train.Checkpoint(step=global_step, epoch=epoch_var, optimizer=opt, model=model)
    #manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=20)

    max_steps_per_episode = 1000

    hidden_size = 128
    batch_size = 512

    replay_buffer_size = 10000
    warmup_sample_steps = 2000
    soft_tau = 5e-3

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
            warmup_sample_steps = warmup_sample_steps,
    )

    max_episode_reward = -100
    rewards = []
    for episode in range(100000):
        initial_state = env.reset()
        initial_state = initial_state.astype(np_dtype)

        episode_reward = sac.run_episode(initial_state)

        episode_reward = episode_reward.numpy()

        rewards.append(episode_reward)
        if len(rewards) > 100:
            rewards = rewards[1:]
        avg_reward = np.mean(rewards)

        q_logs = [f'q{q_idx}: {q.result():2.4f}' for q_idx, q in enumerate(sac.q_metrics)]
        q_log = ', '.join(q_logs)

        logger.info(f'{episode:4d}: step: {sac.global_step.numpy():5d}, episode_reward: {episode_reward:5.2f}/{max_episode_reward:5.2f}, avg_reward: {avg_reward:5.2f}, losses: '
                f'policy: {sac.policy_metric.result():3.4f}, '
                f'{q_log}, '
                f'alpha: {sac.alpha_metric.result():2.4f}, '
                f'entropy: {tf.convert_to_tensor(sac.alpha).numpy():.4f}')

        if episode_reward > max_episode_reward:
            max_episode_reward = episode_reward

            done = False
            state = env.reset()

            step = 0
            while not done:
                actions = sac.policy.get_action(state)
                state, reward, done, _ = env.step(actions)
                env.render(episode, step)
                step += 1

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return actions

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return actions

if __name__ == "__main__":
    #env = NormalizedActions(gym.make("Pendulum-v0"))
    #env = gym.make("Pendulum-v0")

    if True:
        import image_map_env
        env = image_map_env.CustomEnv('maps/map_simple0.png')

    seed = 42
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    main(env)
