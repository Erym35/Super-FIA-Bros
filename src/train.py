import neat
import os

class Train:
    def __init__(self, generations, parallel, level):
        self.generations = generations
        self.level = level
        
    def _run(self, config_file):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        winner = p.run(self._eval_genomes, self.generations)
        
    def _eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 0
            
    def main(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self._run(config_path)

if __name__ == "__main__":
    t = Train(100, 1, '1-1')
    t.main()
