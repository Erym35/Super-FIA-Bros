import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import pickle
import cv2

def main():
    actions = pickle.load(open('src/best_actions.pkl', 'rb'))
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.reset()
    
    video = cv2.VideoWriter('replay.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (256, 240))
    
    for action in actions:
        state, _, done, _ = env.step(action)
        video.write(cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
        if done: break
        
    video.release()
    env.close()

if __name__ == "__main__":
    main()
