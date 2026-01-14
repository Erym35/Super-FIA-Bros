import neat
import pickle
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import cv2

def main(config_file, genome_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    genome = pickle.load(open(genome_file, 'rb'))
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    state = env.reset()
    done = False
    
    while not done:
        state = cv2.resize(state, (13, 16))
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = state.flatten()
        output = net.activate(state)
        action = output.index(max(output))
        state, reward, done, info = env.step(action)
        env.render()
        
    env.close()

if __name__ == "__main__":
    main('config', 'winner.pkl')
