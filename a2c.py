from typing import *

import gym
import tqdm


import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


eps = np.finfo(np.float32).eps.item()

class Actor(tf.keras.Model):
    def __init__(self, hidden_size: int, num_actions: int):
        super().__init__()

        self.dense0 = tf.keras.layers.Dense(hidden_size)
        self.dense1 = tf.keras.layers.Dense(num_actions)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.dense0(inputs)
        x = tf.nn.relu(x)
        x = self.dense1(x)
        return x

class Critic(tf.keras.Model):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.dense0 = tf.keras.layers.Dense(hidden_size)
        self.dense1 = tf.keras.layers.Dense(1)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.dense0(inputs)
        x = tf.nn.relu(x)
        x = self.dense1(x)
        return x

class ActorCritic(tf.keras.Model):
    def __init__(self, hidden_size: int, num_actions: int):
        super().__init__()

        self.actor = Actor(hidden_size, num_actions)
        self.critic = Critic(hidden_size)

    def __call__(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        value = self.critic(inputs)
        policy_dist = self.actor(inputs)

        return policy_dist, value

class ActorCriticShared(tf.keras.Model):
    def __init__(self, hidden_size: int, num_actions: int):
        super().__init__()

        self.dense_common = tf.keras.layers.Dense(hidden_size)
        self.actor = tf.keras.layers.Dense(num_actions)
        self.critic = tf.keras.layers.Dense(1)

    def __call__(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.dense_common(inputs)
        x = tf.nn.relu(x)

        policy_dist = self.actor(x)
        value = self.critic(x)

        return policy_dist, value

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""

    state, reward, done, _ = env.step(action)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])

def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs_all = tf.TensorArray(dtype=initial_state.dtype, size=0, dynamic_size=True)
    values_all = tf.TensorArray(dtype=initial_state.dtype, size=0, dynamic_size=True)
    rewards_all = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        state = tf.expand_dims(state, 0)

        prob_logits, value = model(state)

        # Sample next action from the action probability distribution
        action = tf.random.categorical(prob_logits, 1)[0, 0]
        action_probs = tf.nn.softmax(prob_logits)

        values_all = values_all.write(t, tf.squeeze(value))
        action_probs_all = action_probs_all.write(t, action_probs[0, action])

        state, reward, done = tf_env_step(action)
        state = tf.cast(state, initial_state.dtype)
        state.set_shape(initial_state_shape)

        rewards_all = rewards_all.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs_all.stack()
    values = values_all.stack()
    rewards = rewards_all.stack()

    return action_probs, values, rewards

def get_expected_return(
        rewards: tf.Tensor,
        gamma: float,
        dtype: tf.dtypes.DType,
        standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=dtype, size=n)

    # Start from the end of `rewards` and accumulate reward sums into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype)
    discounted_sum = tf.constant(0, dtype=dtype)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)

    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / 
                   (tf.math.reduce_std(returns) + eps))

    return returns

class ActorCriticLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    @tf.function
    def __call__(self,
            action_probs: tf.Tensor,
            values: tf.Tensor,
            returns: tf.Tensor) -> tf.Tensor:
        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)
        critic_loss = tf.cast(critic_loss, values.dtype)

        return actor_loss + critic_loss

@tf.function
def train_step(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_obj: tf.keras.losses.Loss,
        gamma: float,
        max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""

    with tf.GradientTape() as tape:
        action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode)

        returns = get_expected_return(rewards, gamma, values.dtype)
        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 
        #action_probs, values, returns = [tf.cast(x, tf.float32) for x in [action_probs, values, returns]] 

        loss = loss_obj(action_probs, values, returns)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)
    return episode_reward

def a2c(env):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    use_fp16 = False
    initial_learning_rate = 1e-2
    min_learning_rate = 1e-5

    dtype = tf.float32
    if use_fp16:
        dtype = tf.float16
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    hidden_size = 64
    actor_critic = ActorCriticShared(hidden_size, num_outputs)
    #actor_critic = ActorCritic(hidden_size, num_outputs)

    global_step = tf.Variable(0, dtype=tf.int64, name='global_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    learning_rate = tf.Variable(initial_learning_rate, dtype=tf.float32, name='learning_rate')
    epoch_var = tf.Variable(0, dtype=tf.float32, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    if False:
        opt = tfa.optimizers.RectifiedAdam(lr=learning_rate, min_lr=min_learning_rate)
        opt = tfa.optimizers.Lookahead(opt, sync_period=6, slow_step_size=0.5)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if use_fp16:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

    loss_obj = ActorCriticLoss()
    #checkpoint = tf.train.Checkpoint(step=global_step, epoch=epoch_var, optimizer=opt, model=model)
    #manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=20)

    max_episodes = 10000
    max_steps_per_episode = 1000

    reward_threshold = 195
    num_reward_episodes = 100
    last_rewards = []

    # Discount factor for future rewards
    gamma = 0.99

    with tqdm.trange(max_episodes) as t:
        for i in t:
            initial_state = tf.constant(env.reset(), dtype=dtype)

            reward = train_step(initial_state, actor_critic, opt, loss_obj, gamma, max_steps_per_episode)
            episode_reward = int(reward)

            last_rewards.append(episode_reward)
            if len(last_rewards) >= num_reward_episodes:
                last_rewards = last_rewards[1:]

            avg_reward = np.mean(last_rewards)

            t.set_description(f'Episode {i}')
            t.set_postfix(episode_reward=episode_reward, running_reward=avg_reward)

            if avg_reward > reward_threshold:
                break

    print(f'\nSolved at episode {i}: average reward: {avg_reward:.2f}!')

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    seed = 42
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    a2c(env)
