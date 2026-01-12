import graphviz
import matplotlib.pyplot as plt
import neat

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    if plt is None:
        return
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if view:
        plt.show()
    plt.savefig(filename)
    plt.close()

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True):
    # Wrapper for neat.visualize (omitted for brevity)
    pass
