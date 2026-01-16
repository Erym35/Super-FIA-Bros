import neat
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
import os
import multiprocessing as mp

class Train:
    def __init__(self, generations, parallel, level):
        self.generations = generations
        self.level = level
        self.par = parallel
        self.actions = SIMPLE_MOVEMENT

    def _fitness_func(self, genome, config, queue):
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
        queue.put((genome.key, info['x_pos']))

    def _eval_genomes(self, genomes, config):
        for i in range(0, len(genomes), self.par):
            output = mp.Queue()
            processes = []
            for genome_id, genome in genomes[i:i+self.par]:
                p = mp.Process(target=self._fitness_func, args=(genome, config, output))
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()
                
            results = [output.get() for _ in processes]
            for genome_id, genome in genomes[i:i+self.par]:
                for key, fitness in results:
                    if key == genome_id:
                        genome.fitness = fitness

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
