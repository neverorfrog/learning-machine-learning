import gymnasium as gym
import numpy as np

class CarEnv():
    
    def __init__(self):
        self.env_arguments = {
            'domain_randomize': False,
            'continuous': False,
            'render_mode': 'state_pixels'
        }
        self.env_name = 'CarRacing-v2'
        self.env = gym.make(self.env_name, **self.env_arguments)
        print("Environment:", self.env_name)
        print("Action space:", self.env.action_space)
        print("Observation space:", self.env.observation_space)
        

    def play(self,model):
        seed = 2000
        obs, _ = self.env.reset(seed=seed)
        
        # drop initial frames
        action0 = 0
        for i in range(50):
            obs,_,_,_,_ = self.env.step(action0)
        
        done = False
        total_reward = 0
        terminated = False
        frames = 0
        while not done:    
            
            obs = np.transpose(obs, (2, 0, 1))
            obs = np.expand_dims(obs, axis = 0)
            action = model.predict(obs).item()
            
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            frames += 1
            
        print(total_reward)
        print(frames)
        