import random

import numpy as np
import torch

from concepts import concept_instances
from custom_env import make_env
from state_observer import StateObserver
from train_model import load_model

# TODO: human play to gather data instead?
if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = load_model("../runs/20230920-221924/models/model_10000000.pt").to(device)
    env = make_env(seed, state_observer=StateObserver)
    env.add_concepts(concept_instances)
    obs, info = env.reset(seed=seed)

    lives = info['lives']
    for _ in range(10000):
        np_obs = np.array(obs)
        tensor_obs = torch.Tensor(np_obs).to(device)
        reshaped_obs = tensor_obs.unsqueeze(0)
        q_values = q_network(reshaped_obs)
        action = q_values.argmax(dim=1).item()
        if random.random() < 0.01: # random action to get varity TODO: check how much varity there is...
            action = random.randint(0, 3)
        if info['lives'] != lives:
            action = 1 # override action to fire to not get duplicate frames
            lives = info['lives']
            if lives == 0:
                print("New episode")
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = next_obs
    
    env.reset()
    env.save_concepts()
    env.close()