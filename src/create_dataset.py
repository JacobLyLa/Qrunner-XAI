import random

import numpy as np
import torch

from concepts import concept_instances
from custom_env import make_env
from train_model import load_model

if __name__ == '__main__':
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = load_model("../runs/20230914-224528/models/model_9899999.pt").to(device)
    env = make_env(0, state_observer=True)
    env.add_concepts(concept_instances)
    obs, info = env.reset(seed=seed)

    lives = info['lives']
    for _ in range(20000):
        np_obs = np.array(obs)
        tensor_obs = torch.Tensor(np_obs).to(device)
        reshaped_obs = tensor_obs.unsqueeze(0)
        q_values = q_network(reshaped_obs)
        action = q_values.argmax(dim=1).item()
        if random.random() < 0.05: # random action to get varity TODO: check how much varity there is...
            action = random.randint(0, 3)
        if info['lives'] != lives:
            lives = info['lives']
            if lives == 0:
                print("New episode")
            action = 1 # automatically fire
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
    
    env.reset()
    env.save_concepts()
    env.close()