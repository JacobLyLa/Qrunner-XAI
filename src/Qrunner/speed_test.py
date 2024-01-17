import random
import time

from tqdm import tqdm

from src.DRL.wrapped_qrunner import make_env
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
    frames = 5000
    sizes = (84, 84*2, 84*3, 84*4)
    results = {'Original Env': {}, 'Wrapped Env': {}}
    
    print("Testing FPS for original env")
    for size in sizes:
        env = QrunnerEnv(render_mode='rgb_array', size=size)
        fps = test_fps(env, frames)
        results['Original Env'][size] = fps
        
    print("Testing FPS for wrapped env")
    for size in sizes:
        env = make_env(render_mode='rgb_array', size=size)
        fps = test_fps(env, frames)
        results['Wrapped Env'][size] = fps
        
    print("\nFPS Results:")
    print(f"{'Env/Size':<15} {'84 px':<8} {'168 px':<8} {'252 px':<8} {'336 px':<8}")
    for env, fps_results in results.items():
        print(f"{env:<15} {fps_results[sizes[0]]:<8.2f} {fps_results[sizes[1]]:<8.2f} {fps_results[sizes[2]]:<8.2f} {fps_results[sizes[3]]:<8.2f}")

if __name__ == "__main__":
    main()

'''
FPS Results:
Env/Size        84 px    168 px   252 px   336 px  
Original Env    4192.83  2498.39  1337.99  996.42  
Wrapped Env     1186.39  687.46   349.00   248.41  
'''