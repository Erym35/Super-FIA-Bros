import os
import sys
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR) 
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import pickle
import cv2
import numpy as np
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser(description='Replay best actions')
    parser.add_argument('--level', type=str, default='1-1', help='Level to replay (e.g. 1-1, 1-2)')
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    actions_path = os.path.join(local_dir, 'best_actions.pkl')
    video_dir = os.path.join(local_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    video_output_path = os.path.join(video_dir, f'mario_replay_{args.level}.mp4')
    
    if not os.path.exists(actions_path):
        print(f"Errore: {actions_path} non trovata.")
        sys.exit(1)
        
    actions = pickle.load(open(actions_path, 'rb'))
    env_name = f'SuperMarioBros-{args.level}-v0'
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    try:
        env.seed(42)
        env.action_space.seed(42)
    except:
        pass
        
    state = env.reset()
    height, width, _ = env.observation_space.shape
    video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
    
    info = {'x_pos': 0}
    
    for action in actions:
        state, reward, done, info = env.step(action)
        frame = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
        
        # HUD
        current_x = info.get('x_pos', 0)
        cv2.putText(frame, f"DIST: {current_x}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        video.write(frame)
        
        if done and info['x_pos'] > 3000:
             print("Victory!")
             break
            
    video.release()
    env.close()
    
    default_output_path = os.path.join(video_dir, 'mario_replay.mp4')
    try:
        shutil.copy(video_output_path, default_output_path)
    except:
        pass

if __name__ == "__main__":
    main()
