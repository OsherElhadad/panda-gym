import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import os

class CustomGoalCallback(BaseCallback):
    def __init__(self, envs, goals, verbose=0):
        super(CustomGoalCallback, self).__init__(verbose)
        self.envs = envs
        self.goals = goals

    def _on_step(self):
        for env, goal in zip(self.envs, self.goals):
            env.set_goal(goal)
        return True

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, model_name, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_name = model_name

        # Record the start time when the callback is initialized
        self.start_time = time.time()

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            save_path = os.path.join(self.save_path, f"{self.model_name}_step_{self.n_calls}.zip")
            self.model.save(save_path)
            print(f"Model saved at step {self.n_calls} to {save_path}")
        return True

class CustomPandaReachEnv(gym.Env):
    def __init__(self, goal, reward_type='dense'):
        super().__init__()
        self.env = gym.make('PandaReachDense-v3', render_mode='rgb_array') if reward_type == 'dense' else gym.make('PandaReach-v3', render_mode='rgb_array')
        self.goal = goal
        self.reward_type = reward_type
        self.env.unwrapped.goal = goal
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def set_goal(self, goal):
        self.goal = goal
        self.env.unwrapped.goal = goal

    def set_reward_type(self, reward_type):
        self.reward_type = reward_type

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == 'dense':
            return -distance
        elif self.reward_type == 'sparse':
            return -(distance > 0.05).astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        obs['desired_goal'] = self.goal
        info['is_success'] = reward == 0 if self.reward_type == 'sparse' else False
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        result = self.env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        self.env.unwrapped.goal = self.goal
        obs['desired_goal'] = self.goal
        return obs, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

def make_env(goal, reward_type='dense'):
    def _init():
        env = CustomPandaReachEnv(goal, reward_type)
        return env
    return _init

def train_and_evaluate(agent_type, reward_type, timesteps, goals, save_dir, n_sampled_goal=None):
    start_time = time.time()
    print(f"Training {agent_type.upper()} agent with {reward_type} rewards for {timesteps} timesteps and {len(goals)} goals.")
    
    # Calculate save frequency based on the number of goals
    total_timesteps = 1000000
    save_freq = total_timesteps // len(goals) // 100
    vec_env = make_vec_env(lambda: CustomPandaReachEnv(goals[0], reward_type), n_envs=len(goals))

    if agent_type == 'sac':
        model = SAC('MultiInputPolicy', vec_env, verbose=1)
    elif agent_type == 'uvfa':
        model = SAC(
            "MultiInputPolicy",
            vec_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy='future',
            ),
            learning_starts=1000,
            verbose=1,
        )

    callback = CustomGoalCallback([CustomPandaReachEnv(goal, reward_type) for goal in goals], goals)
    model_name = f"{agent_type}-{len(goals)}g-PandaReach-{reward_type}"
    save_callback = SaveModelCallback(save_freq=save_freq, save_path=save_dir, model_name=model_name)
    print(f"Starting training for {timesteps} timesteps...")
    
    model.learn(total_timesteps=timesteps, callback=[callback, save_callback])
    
    final_model_name = f"{model_name}-{timesteps // 1000}k"
    model.save(f"{save_dir}/{final_model_name}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_file_path = os.path.join(save_dir, f"{final_model_name}_final_time.txt")
    with open(time_file_path, 'w') as time_file:
        time_file.write(f"Total elapsed time: {elapsed_time} seconds\n")
    print(f"Training completed for {timesteps} timesteps. Final model saved as {final_model_name}. Total elapsed time: {elapsed_time} seconds")
    return elapsed_time

def train_single_goal(agent_type, reward_type, timesteps, save_dir):
    start_time = time.time()
    print(f"Training {agent_type.upper()} agent with {reward_type} rewards for {timesteps} timesteps on a single goal.")
    single_goal = np.random.uniform(low=[0.3, 0.1, 0.1], high=[0.5, 0.3, 0.3])
    np.save(os.path.join(save_dir, 'single_goal.npy'), single_goal)
    env = CustomPandaReachEnv(single_goal, reward_type)

    print(f"Setting up {agent_type.upper()} model...")
    model = SAC('MultiInputPolicy', env, verbose=1)
    model_name = f"{agent_type}-single-PandaReach-{reward_type}"
    save_callback = SaveModelCallback(save_freq=2000, save_path=save_dir, model_name=model_name)
    print(f"Starting training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=save_callback)
    model.save(f"{save_dir}/{model_name}-{timesteps // 1000}k")
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_file_path = os.path.join(save_dir, f"{model_name}_final_time.txt")
    with open(time_file_path, 'w') as time_file:
        time_file.write(f"Total elapsed time: {elapsed_time} seconds\n")
    print(f"Training completed for {timesteps} timesteps. Final model saved as {model_name}-{timesteps // 1000}k. Total elapsed time: {elapsed_time} seconds")
    return elapsed_time

def generate_and_save_goals(num_goals, filename):
    goals = np.random.uniform(low=[0.3, 0.1, 0.1], high=[0.5, 0.3, 0.3], size=(num_goals, 3))
    np.save(filename, goals)
    return goals

goal_files = {
    1: 'goals_1.npy',
    5: 'goals_5.npy',
    10: 'goals_10.npy',
    15: 'goals_15.npy',
    20: 'goals_20.npy',
    25: 'goals_25.npy',
    30: 'goals_30.npy'
}

# Generate and save goals
for num_goals, filename in goal_files.items():
    if not os.path.exists(filename):
        generate_and_save_goals(num_goals, filename)

# Load goals
goal_sets = {num_goals: np.load(filename) for num_goals, filename in goal_files.items()}

timesteps_list = [1000000]
save_dir = "./models_sparse_sac"
training_times = []

# Train and evaluate multi-goal agents
for reward_type in ['sparse', 'dense']:
    for agent_type in ['sac', 'uvfa']:
        for num_goals, goals in goal_sets.items():
            print(f"Training with {num_goals} goals")
            if agent_type == 'uvfa':
                n_sampled_goal = len(goals)
                for timesteps in timesteps_list:
                    elapsed_time = train_and_evaluate(agent_type, reward_type, timesteps, goals, save_dir, n_sampled_goal)
            else:
                for timesteps in timesteps_list:
                    elapsed_time = train_and_evaluate(agent_type, reward_type, timesteps, goals, save_dir)
            training_times.append((f"{agent_type.upper()} with {num_goals} goals and {reward_type} rewards", elapsed_time))

# Train and evaluate single-goal agent with dense rewards
single_goal_time = train_single_goal('sac', 'dense', 1000000, save_dir)
training_times.append(("Single goal SAC dense agent", single_goal_time))

# Train and evaluate single-goal agent with sparse rewards
single_goal_time = train_single_goal('sac', 'sparse', 1000000, save_dir)
training_times.append(("Single goal SAC sparse agent", single_goal_time))
