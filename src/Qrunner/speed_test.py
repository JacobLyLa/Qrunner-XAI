# TODO: move to a notebook as it is not core? (not being imported by other files)

import random
import time

from tqdm import tqdm

from src.DRL.wrapped_qrunner import wrapped_qrunner_env
from src.Qrunner.qrunner import QrunnerEnv


def test_fps(env, num_frames):
    start = time.time()
    obs, info = env.reset()
    for _ in tqdm(range(num_frames), desc="Processing frames"):
        action = 3 if random.random() < 0.2 else 2 # move right and sometimes jump
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    end = time.time()
    return num_frames / (end - start)

def main():
    frames = 10000
    scales = (84, 84*2, 84*3, 84*4)
    results = {'Original Env': {}, 'Wrapped Env': {}}
    
    print("Testing FPS for original env")
    # TODO: Normal env doesnt use scale anymore, only need to test once
    for scale in scales:
        env = QrunnerEnv()
        fps = test_fps(env, frames)
        results['Original Env'][scale] = fps
        
    print("Testing FPS for wrapped env")
    for scale in scales:
        env = wrapped_qrunner_env(frame_skip=1, scale=scale)
        fps = test_fps(env, frames)
        results['Wrapped Env'][scale] = fps
        
    print("\nFPS Results:")
    print(f"{'Env/Scale':<15} {'84 px':<8} {'168 px':<8} {'252 px':<8} {'336 px':<8}")
    for env, fps_results in results.items():
        print(f"{env:<15} {fps_results[scales[0]]:<8.2f} {fps_results[scales[1]]:<8.2f} {fps_results[scales[2]]:<8.2f} {fps_results[scales[3]]:<8.2f}")

if __name__ == "__main__":
    main()

'''
FPS Results:
Env/Scale        84 px    168 px   252 px   336 px  
Original Env    4192.83  2498.39  1337.99  996.42  
Wrapped Env     1186.39  687.46   349.00   248.41  
'''