import neat
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
import os

class Train:
    def __init__(self, generations, parallel, level):
        self.generations = generations
        self.level = level
        self.actions = SIMPLE_MOVEMENT

    def _run_single_episode(self, genome, config):
        env = gym_super_mario_bros.make('SuperMarioBros-'+self.level+'-v0')
        env = JoypadSpace(env, self.actions)
        state = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        done = False
        info = {'x_pos': 0}
        
        while not done:
            state = cv2.resize(state, (13, 16))
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = state.flatten()
            output = net.activate(state)
            action = output.index(max(output))
            state, reward, done, info = env.step(action)
            
        env.close()
        return info['x_pos']

    def _eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self._run_single_episode(genome, config)

    def _run(self, config_file):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        p.run(self._eval_genomes, self.generations)

    def main(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self._run(config_path)
