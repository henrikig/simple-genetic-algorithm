import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from GeneticAlgorithm import GeneticAlgorithm


class SineGA(GeneticAlgorithm):
    """Sub-class implementation of the genetic algorithm with sine fitness function"""

    def __init__(
        self,
        population_size=100,
        chromosome_length=15,
        crossover_rate=0.6,
        mutation_rate=0.00333,
        selection_method="roulette",
        crowding=False,
        elitism=0,
        tournament_size=3,
        max_iter=1000,
        plot_iter=None,
    ):
        super().__init__(
            population_size,
            chromosome_length,
            crossover_rate,
            mutation_rate,
            selection_method,
            crowding,
            elitism,
            tournament_size,
            max_iter,
            plot_iter,
        )

        # Calculate scale which is applied to get fitness values on [0, 128]
        self.scale = self.chromosome_length - 7

    def on_fitness(self):
        """Calculates the fitness of the entire population"""
        self.fitness = [1 + np.sin(int(bitstring, 2) * 2 ** (-self.scale)) for bitstring in self.population]
        self.objective = [-fitness for fitness in self.fitness]

    def individual_fitness(self, bitstring):
        """Calculates fitness of an individual"""
        return 1 + np.sin(int(bitstring, 2) * 2 ** (-self.scale))

    def is_finished(self) -> bool:
        """Termination criteria for the sine optimization"""
        return min(self.fitness) > 1.9

    def plot(self, title=None):
        """Plots individuals in a generation with the sine wave"""
        x = np.arange(0, 128, 0.1)
        y = np.sin(x)

        scale = self.chromosome_length - 7
        x_fitness = np.array([int(bitstring, 2) * 2 ** (-scale) for bitstring in self.population])
        y_fitness = np.sin(x_fitness)
        plt.plot(x, y)
        plt.plot(x_fitness, y_fitness, "o", color="orange")
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_entropy(self):
        """Plots the entropy of last two recent runs of crowding vs no crowding"""
        with Path("data/sine_crowding.p").open("rb") as fp:
            crowding = pickle.load(fp)
        with Path("data/sine_no_crowding.p").open("rb") as fp:
            no_crowding = pickle.load(fp)
        plt.plot(crowding, label="Crowding")
        plt.plot(no_crowding, color="darkorange", label="No Crowding")
        plt.title("Entropy")
        plt.legend(loc="upper right")
        plt.show()

    def save_entropy(self):
        """Stores the entropy for later use"""
        if self._crowding:
            with Path("data/sine_crowding.p").open("wb") as fp:
                pickle.dump(self._entropy, fp)
        else:
            with Path("data/sine_no_crowding.p").open("wb") as fp:
                pickle.dump(self._entropy, fp)


if __name__ == "__main__":
    ga = SineGA(
        population_size=100,
        chromosome_length=15,
        crossover_rate=0.6,
        mutation_rate=0.00333,
        selection_method="tournament",
        crowding=True,
        elitism=4,
        tournament_size=10,
        max_iter=1000,
        plot_iter=[1, 5, 10, 20, 50],
    )
    ga.main()
