from GeneticAlgorithm import GeneticAlgorithm
from LinReg import LinReg
import pandas as pd
import matplotlib.pyplot as plt
import pickle


class LinRegGA(GeneticAlgorithm):
    """Sub-class implementation of the genetic algorithm for feature selection"""
    def __init__(self, population_size=100, chromosome_length=15, crossover_rate=0.6,
                 mutation_rate=0.0333, selection_method='roulette', crowding=False,
                 elitism=0, tournament_size=3, max_iter=1000, plot_iter=None):
        super().__init__(population_size, chromosome_length, crossover_rate, mutation_rate,
                         selection_method, crowding, elitism, tournament_size, max_iter, plot_iter)

        # Initiate LinReg class and data for fitness calculation
        self.lr = LinReg()
        self.df = pd.read_csv('dataset.csv', header=None)
        self.data = self.df[self.df.columns[:-1]]
        self.values = self.df[self.df.columns[-1]]
        self.basic_fitness = self.lr.get_fitness(self.data, self.values)

        self.plot_data = []

    def on_fitness(self):
        """Calculates the relative fitness of the entire population"""
        self.objective = [self.lr.get_fitness(self.lr.get_columns(self.data, bitstring), self.values)
                          for bitstring in self.population]
        fitness_hi = max(self.objective)
        fitness_lo = min(self.objective)
        diff = fitness_hi - fitness_lo
        self.fitness = [(fitness_hi - fitness) + diff for fitness in self.objective]
        self.plot_data.append(fitness_lo)
        print(len(self.population))
        print(fitness_lo)

    def individual_fitness(self, bitstring):
        """Calculates fitness of an individual"""
        return -self.lr.get_fitness(self.lr.get_columns(self.data, bitstring), self.values)

    def is_finished(self) -> bool:
        """Termination criteria for the feature selection optimization"""
        return min(self.objective) < 0.123

    def plot(self, title=None):
        """Plots the fitness improvements over the generations compared to baseline fitness"""
        plt.plot(self.plot_data, color="darkorange", label="Fitness")
        plt.axhline(y=self.basic_fitness, label="Baseline")
        plt.title("Fitness improvement")
        plt.legend(loc="upper right")
        plt.show()

    def plot_entropy(self):
        """Plots the entropy of last two recent runs of crowding vs no crowding"""
        with open("data/linreg_crowding.p", "rb") as fp:
            crowding = pickle.load(fp)
        with open("data/linreg_no_crowding.p", "rb") as fp:
            no_crowding = pickle.load(fp)
        plt.plot(crowding, label="Crowding")
        plt.plot(no_crowding, color="darkorange", label="No Crowding")
        plt.title("Entropy")
        plt.legend(loc="upper right")
        plt.show()

    def save_entropy(self):
        """Stores the entropy to be used for plotting"""
        if self._crowding:
            with open('data/linreg_crowding.p', 'wb') as fp:
                pickle.dump(self._entropy, fp)
        else:
            with open('data/linreg_no_crowding.p', 'wb') as fp:
                pickle.dump(self._entropy, fp)


if __name__ == '__main__':
    ga = LinRegGA(population_size=100, chromosome_length=101, crossover_rate=0.6, mutation_rate=0.0333,
                  selection_method='tournament', crowding=False, elitism=6, tournament_size=10, max_iter=100)
    ga.main()
