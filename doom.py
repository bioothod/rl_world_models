import vizdoom

import numpy as np
import tensorflow as tf

from a2c import a2c

def run(env, actions):

def main():
    game = DoomGame()

    # This corresponds to the simple task we will pose our agent
    game.set_doom_scenario_path("basic.wad")
    game.set_doom_map("map01")
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_160X120)
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.add_available_button(vizdoom.Button.MOVE_LEFT)
    game.add_available_button(vizdoom.Button.MOVE_RIGHT)
    game.add_available_button(vizdoom.Button.ATTACK)
    game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
    game.add_available_game_variable(vizdoom.GameVariable.POSITION_X)
    game.add_available_game_variable(vizdoom.GameVariable.POSITION_Y)
    game.set_episode_timeout(300)
    game.set_episode_start_time(10)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_living_reward(-1)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.init()

    actions = np.identity(a_size,dtype=bool).tolist()
    env = game

    seed = 42
    env.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    initial_learning_rate = 1e-2
    min_learning_rate = 1e-5

    dtype = tf.float32

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

    for i in range(max_episodes):
        env.new_episode()
        initial_state = env.get_state().screen_buffer

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
    run(env, actions)

if __name__ == "__main__":
    main()
