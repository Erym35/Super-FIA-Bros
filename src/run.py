import neat
import os
import warnings
warnings.filterwarnings("ignore")
import pickle
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import visualize
import cv2

ACTIONS = SIMPLE_MOVEMENT

def main(config_file, file, level="1-1"):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    genome = pickle.load(open(file, 'rb'))
    env = gym_super_mario_bros.make('SuperMarioBros-'+level+'-v0')
    env = JoypadSpace(env, ACTIONS)
    
    video_dir = os.path.join(os.path.dirname(__file__), 'videos')
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, 'mario_run.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    height, width, _ = env.observation_space.shape
    video = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    state = env.reset()
    done = False
    
    while not done:
        state_input = cv2.resize(state, (13, 16))
        state_input = cv2.cvtColor(state_input, cv2.COLOR_RGB2GRAY)
        state_input = state_input.flatten()
        
        output = net.activate(state_input)
        ind = output.index(max(output))
        state, reward, done, info = env.step(ind)
        
        video.write(cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
        
    video.release()
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="1-1")
    args = parser.parse_args()
    main('config', "winner.pkl", args.level)
