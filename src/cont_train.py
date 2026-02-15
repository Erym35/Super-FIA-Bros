import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import neat
import pickle
import gym
import logging
logging.getLogger("gym").setLevel(logging.ERROR)
import multiprocessing as mp
import visualize
import train

gym.logger.set_level(40)

# FIX: Register custom activations and mock missing modules for old checkpoints
import sys
import types
from neat.activations import ActivationFunctionSet

# 1. Custom Activations
def elu_activation(z):
    return z if z > 0.0 else (np.exp(z) - 1)

def lelu_activation(z):
    leaky = 0.005
    return z if z > 0.0 else leaky * z

def selu_activation(z):
    lam = 1.0507
    alpha = 1.67326
    return lam * z if z > 0.0 else lam * alpha * (np.exp(z) - 1)

try:
    import neat.activations
    for name, func in [('elu_activation', elu_activation), 
                       ('lelu_activation', lelu_activation), 
                       ('selu_activation', selu_activation)]:
        if not hasattr(neat.activations, name):
            setattr(neat.activations, name, func)
except Exception as e:
    print(f"Warning: Could not register custom activations: {e}")

# 2. Mock missing modules (e.g. neat.innovation used in some custom NEAT forks)
try:
    if 'neat.innovation' not in sys.modules:
        mock_innovation = types.ModuleType('neat.innovation')
        sys.modules['neat.innovation'] = mock_innovation
        # Add 'Innovation' class if it was used in pickle
        class MockInnovation:
            pass
        mock_innovation.Innovation = MockInnovation
        
        # Add 'InnovationTracker' class (Common in some NEAT forks)
        class MockInnovationTracker:
            pass
        mock_innovation.InnovationTracker = MockInnovationTracker
        
    if 'neat.genes' not in sys.modules:
        # Some old versions/forks separate genes differently
        import neat.genome
        sys.modules['neat.genes'] = neat.genome
        
except Exception as e:
    print(f"Warning: Could not mock missing modules: {e}")


import copy
import signal
import sys

class Train(train.Train):
    def __init__(self, generations, file_name, parallel, level):
        super().__init__(generations, parallel, level)
        self.actions = train.SIMPLE_MOVEMENT
        self.lock = mp.Lock()
        self.file_name = file_name

    def _run(self, config_file, n):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        # p = neat.Population(config)
        p = neat.Checkpointer.restore_checkpoint(self.file_name)
        
        # SAFELY UPDATE CONFIG:
        # Instead of replacing the entire config object (which resets node indexers and causes AssertionErrors),
        # we specifically update the parameters we care about from the loaded config file.
        
        # 1. Species Set Config (Diversity)
        p.config.species_set_config.compatibility_threshold = config.species_set_config.compatibility_threshold
        
        # 2. Stagnation Config
        p.config.stagnation_config.max_stagnation = config.stagnation_config.max_stagnation
        p.config.stagnation_config.species_fitness_func = config.stagnation_config.species_fitness_func
        p.config.stagnation_config.species_elitism = config.stagnation_config.species_elitism
        
        # 3. Genome Config (Mutation rates, etc.)
        # We update the mutation probabilities but MUST NOT touch the indexers
        p.config.genome_config.conn_add_prob = config.genome_config.conn_add_prob
        p.config.genome_config.conn_delete_prob = config.genome_config.conn_delete_prob
        p.config.genome_config.node_add_prob = config.genome_config.node_add_prob
        p.config.genome_config.node_delete_prob = config.genome_config.node_delete_prob
        p.config.genome_config.weight_mutate_rate = config.genome_config.weight_mutate_rate
        p.config.genome_config.bias_mutate_rate = config.genome_config.bias_mutate_rate
        p.config.genome_config.activation_mutate_rate = config.genome_config.activation_mutate_rate
        
        # 4. Reproduction Config (Elitism & Survival) -> ESSENTIAL FOR CONVERGENCE TWEAK
        p.config.reproduction_config.elitism = config.reproduction_config.elitism
        p.config.reproduction_config.survival_threshold = config.reproduction_config.survival_threshold
        
        # Print confirmation
        print(f"Updated config from file: Compat Thresh={p.config.species_set_config.compatibility_threshold}, Max Stag={p.config.stagnation_config.max_stagnation}")
        print(f"VERIFIED ELITISM: {p.config.reproduction_config.elitism}, SURVIVAL: {p.config.reproduction_config.survival_threshold}")
        print(f"VERIFIED MUTATION: Conn Add={p.config.genome_config.conn_add_prob}, Weight Mutate={p.config.genome_config.weight_mutate_rate}")

        p.add_reporter(neat.StdOutReporter(True))
        
        # Checkpoint to specific folder
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        p.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(checkpoint_dir, 'neat-checkpoint-')))
        
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        print("loaded checkpoint...")
        
        try:
            winner = p.run(self._eval_genomes, n)
            win = p.best_genome
            pickle.dump(winner, open('winner.pkl', 'wb'))
            pickle.dump(win, open('real_winner.pkl', 'wb'))
    
            visualize.draw_net(config, winner, True)
            visualize.plot_stats(stats, ylog=False, view=False)
            visualize.plot_species(stats, view=False)
        except KeyboardInterrupt:
            # Ignore further interrupts to allow saving to complete
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except KeyboardInterrupt:
                pass # Already interrupted, just proceed
            
            print("\nUser interrupted! Saving best genome so far...")
            
            if self.best_genome is not None:
                win = self.best_genome
            elif p.best_genome is not None:
                win = p.best_genome
            else:
                win = None

            if win is not None:
                # IMPORTANT: Deepcopy to prevent reference issues if used later
                win_copy = copy.deepcopy(win)
                pickle.dump(win_copy, open('winner.pkl', 'wb'))
                pickle.dump(win_copy, open('real_winner.pkl', 'wb'))
                print(f"Best genome (Fitness: {win.fitness}) saved to winner.pkl")
                
                try:
                    visualize.draw_net(config, win, True)
                    visualize.plot_stats(stats, ylog=False, view=False)
                    visualize.plot_species(stats, view=False)
                    print("Graphs generated on interrupt.")
                except Exception as e:
                    print(f"Graph generation failed: {e}")
            else:
                print("No genome found to save.")
            
            sys.exit(0)


if __name__ == "__main__":
    # Example: python src/cont_train.py (User needs to edit path manually or pass as arg)
    checkpoint_to_load = "./src/checkpoints/neat-checkpoint-2492" # Placeholder
    
    # Use max available cores
    cores = mp.cpu_count()
    print(f"Using {cores} parallel processes.")
