import os
import warnings
# Suppress all Gym/Deprecation warnings aggressively
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import logging
logging.getLogger("gym").setLevel(logging.ERROR)
import neat
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import cv2
import numpy as np
import signal
import sys
import multiprocessing as mp
import pickle
import visualize
import copy

gym.logger.set_level(40)

class Train:
    def __init__(self, generations, parallel, level):
        self.generations = generations
        self.lock = None
        self.par = parallel
        self.level = level
        self.actions = SIMPLE_MOVEMENT
        self.best_genome = None

    def _get_actions(self, a):
        return a.index(max(a))

    def _run_single_episode(self, genome, config, seed):
        env = gym_super_mario_bros.make('SuperMarioBros-'+self.level+'-v0')
        env = JoypadSpace(env, self.actions)
        try:
            env.seed(seed)
            env.action_space.seed(seed)
            if hasattr(env.unwrapped, 'seed'):
                try:
                    env.unwrapped.seed(seed)
                except:
                    pass
            
            state = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            done = False
            i = 0
            old = 40
            
            info = {'x_pos': 0}
            
            while not done:
                state = cv2.resize(state, (13, 16))
                state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                state = state.flatten()
                
                output = net.activate(state)
                action_index = self._get_actions(output)
                
                info['actions'] = info.get('actions', []) + [action_index]
                
                s, reward, done, info_step = env.step(action_index)
                
                info.update(info_step)
                state = s
                i += 1
                if i % 100 == 0:
                    if old == info['x_pos']:
                        break
                    else:
                        old = info['x_pos']
                        
            x_pos = int(info['x_pos'])
            fitness = -1 if x_pos <= 40 else x_pos
            
            if fitness > 315: 
                self._check_and_save_best(fitness, info.get('actions', []))
                
            env.close()
            did_win = info.get('flag_get', False) or info.get('x_pos', 0) >= 3161
            game_score = info.get('score', 0)
            return fitness, game_score, did_win

        except KeyboardInterrupt:
            env.close()
            exit()
        except Exception as e:
            print(f"Errore durante la simulazione (seed={seed}): {e}")
            env.close()
            return -1, 0, False

    def _check_and_save_best(self, fitness, actions):
         if fitness > 315:
            best_fitness_path = os.path.join(os.path.dirname(__file__), 'best_fitness.txt')
            saved_actions_path = os.path.join(os.path.dirname(__file__), 'best_actions.pkl')
            
            previous_best = 0.0
            if os.path.exists(best_fitness_path):
                try:
                    with open(best_fitness_path, 'r') as f:
                        previous_best = float(f.read().strip())
                except:
                    pass

            if fitness > previous_best:
                try:
                    print(f"Nuova distanza raggiunta! ({fitness} > {previous_best})!")
                    pickle.dump(actions, open(saved_actions_path, 'wb'))
                    with open(best_fitness_path, 'w') as f:
                        f.write(str(fitness))
                except Exception as e:
                    print(f"Errore salvando la migliore run: {e}")

    def _fitness_func(self, genome, config, o):
        fitness, score, did_win = self._run_single_episode(genome, config, seed=42)
        o.put((genome.key, fitness, score, did_win))

    def _eval_genomes(self, genomes, config):
        idx, genomes = zip(*genomes)
        gen_wins = 0
        gen_total_score = 0
        total_genomes = len(genomes)

        for i in range(0, len(genomes), self.par):
            output = mp.Queue()
            batch = genomes[i:i + self.par]
            processes = [mp.Process(target=self._fitness_func, args=(genome, config, output)) for genome in batch]
            [p.start() for p in processes]
            try:
                [p.join() for p in processes]
            except KeyboardInterrupt:
                [p.terminate() for p in processes]
                raise
            results = [output.get() for _ in processes]
            results_map = {k: (v, s, w) for k, v, s, w in results}

            for genome in batch:
                if genome.key not in results_map:
                    print(f"Warning: Genome {genome.key} missing from results! Assigning default fitness.")
                    genome.fitness = -1.0
                    continue
                
                fitness, score, did_win = results_map[genome.key]
                if fitness is None or not isinstance(fitness, (int, float)):
                    print(f"Warning: Genome {genome.key} has invalid fitness {fitness}. Assigning -1.0.")
                    genome.fitness = -1.0
                else:
                    genome.fitness = fitness
                
                if did_win: gen_wins += 1
                gen_total_score += score
                if self.best_genome is None or (genome.fitness is not None and genome.fitness > self.best_genome.fitness):
                    self.best_genome = copy.deepcopy(genome)
                print(f"Genome {genome.key}: Fit={genome.fitness} | Score={score} | Win={'YES' if did_win else 'NO'}")

        win_rate = (gen_wins / total_genomes) * 100
        avg_score = gen_total_score / total_genomes
        print(f"📊 GENERATION STATS 📊 - Win Rate {win_rate:.2f}% | Avg Score {avg_score:.2f}")

    def _run(self, config_file, n):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        config.reproduction_config.min_species_size = 1
        config.species_set_config.compatibility_threshold = 4.0
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        p.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(checkpoint_dir, 'neat-checkpoint-')))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        
        try:
            winner = p.run(self._eval_genomes, n)
            pickle.dump(winner, open('winner.pkl', 'wb'))
            visualize.draw_net(config, winner, True)
            visualize.plot_stats(stats, ylog=False, view=False)
        except KeyboardInterrupt:
            print("Interrupted! Saving best genome...")
            if self.best_genome: pickle.dump(self.best_genome, open('winner.pkl', 'wb'))
            sys.exit(0)

    def main(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self._run(config_path, self.generations)

if __name__ == "__main__":
    cores = mp.cpu_count()
    t = Train(1000, parallel=cores, level='1-1')
    t.main()
