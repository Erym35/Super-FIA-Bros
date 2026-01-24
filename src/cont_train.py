import train
import neat
import pickle
import os
import multiprocessing as mp

class Train(train.Train):
    def __init__(self, generations, file_name, parallel, level):
        super().__init__(generations, parallel, level)
        self.file_name = file_name

    def _run(self, config_file, n):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Checkpointer.restore_checkpoint(self.file_name)
        p.add_reporter(neat.StdOutReporter(True))
        winner = p.run(self._eval_genomes, n)

if __name__ == "__main__":
    cores = mp.cpu_count()
    t = Train(1000, "neat-checkpoint-0", cores, '1-1')
    t.main()
