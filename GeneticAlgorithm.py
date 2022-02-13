import random
import numpy as np
from math import log


class GeneticAlgorithm:
    """Super class genetic algorithm"""
    def __init__(self, population_size=100, chromosome_length=15,
                 crossover_rate=0.6, mutation_rate=0.00333,
                 selection_method='roulette', crowding=False,
                 elitism=0, tournament_size=3, max_iter=1000, plot_iter=None):

        # Set input variables
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self._crowding = crowding
        self._elitism = elitism
        self.tournament_size = tournament_size
        self.max_iter = max_iter
        if plot_iter is not None:
            self.plot_iter = plot_iter
        else:
            self.plot_iter = []

        # Initiate variables for later use
        self.population = []
        self.mating_pool = []
        self.fitness = []
        self.objective = []
        self.mating_fitness = []
        self._entropy = []
        self.iter_count = 0

        # Initialize population of chromosomes with length 'self.chromosome_length'
        for _ in range(self.population_size):
            self.population.append(f'{random.getrandbits(self.chromosome_length):0{self.chromosome_length}b}')

        # Choose selection method based on input. Defaults to 'roulette'.
        if selection_method == 'roulette':
            self.parent_selection = self.roulette_selection
        elif selection_method == 'tournament':
            self.parent_selection = self.tournament_selection
        else:
            raise ValueError('Not a valid selection method. Please choose "roulette" or "stochastic".')

    # Sub-class based fitness implementations
    def on_fitness(self) -> list:
        pass

    def individual_fitness(self, bitstring) -> float:
        pass

    def roulette_selection(self):
        """Selects parents for mating through fitness proportionate roulette wheel selection"""
        # Calculate total fitness of population
        total_fitness = sum(self.fitness)
        # Calculate proportional fitness for each individual
        proportions = [fitness / total_fitness for fitness in self.fitness]
        # Stochastically pick a parent based on its fitness
        while len(self.mating_pool) < len(self.population) - self._elitism:
            self.mating_pool.append(self.population[np.random.choice(len(self.population), p=proportions)])

    def tournament_selection(self):
        """Selects parents based on tournament outcomes"""
        while len(self.mating_pool) < len(self.population) - self._elitism:
            # Get index for 'k' random parents
            random_parents = random.choices(range(len(self.population)), k=self.tournament_size)
            # Get fitness for the corresponding individuals
            fitness = [self.fitness[i] for i in random_parents]
            # Find the index of the maximum fitness element
            best_index = fitness.index(max(fitness))
            # Add the winner of the tournament to the mating pool
            self.mating_pool.append(self.population[random_parents[best_index]])

    def crossover(self):
        # Iterate through two and two parents in the mating pool
        for i, (p1, p2) in enumerate(zip(self.mating_pool[::2], self.mating_pool[1::2])):
            # Do crossover with probability 'self.crossover_rate'
            if random.random() <= self.crossover_rate:
                # Create random crossover point 'k' and two corresponding offsprings
                k = random.choice(range(1, self.chromosome_length))
                c1 = p1[:k] + p2[k:]
                c2 = p2[:k] + p1[k:]
                # Update mating pool with new chromosomes
                self.mating_pool[2 * i] = c1
                self.mating_pool[2 * i + 1] = c2

    def mutation(self):
        # Iterate all offspring in mating pool
        for i, bitstring in enumerate(self.mating_pool):
            bit_list = list(bitstring)
            # For each bit, mutate (flip) it with a probability of 'self.mutation_rate'
            bit_list = [str(int(bit) ^ 1) if random.random() < self.mutation_rate else bit for bit in bit_list]
            bitstring = ''.join(bit_list)
            # Replace with the mutated bit
            self.mating_pool[i] = bitstring

    def elitism(self):
        # Sort parents based on their fitness
        top_elements = [parent for _, parent in
                        sorted(zip(self.fitness, self.population), reverse=True)]
        # Pick the 'self._elitism' best parents for the next generation
        top_elements = top_elements[:self._elitism]
        # Add selected parents to the mating pool
        self.mating_pool.extend(top_elements)

    def survivor_selection(self):
        """Age based survival where all offspring goes to the next generation"""
        self.population = self.mating_pool.copy()
        self.mating_pool = []

    def crowding(self):
        """Selects individuals to survive based on genotype distance metrics"""
        total_fitness = sum(self.fitness)
        proportions = [fitness / total_fitness for fitness in self.fitness]
        selected_parents = random.choices(range(len(self.population)),
                                          proportions, k=self.population_size-self._elitism)
        for p1_index, p2_index in zip(selected_parents[::2], selected_parents[1::2]):
            p1 = self.population[p1_index]
            p2 = self.population[p2_index]
            p1_fitness = -self.objective[p1_index]
            p2_fitness = -self.objective[p2_index]
            # Create random crossover point 'k'
            k = random.choice(range(1, self.chromosome_length))
            # Do 1-point crossover
            c1 = p1[:k] + p2[k:]
            c2 = p2[:k] + p1[k:]
            children = [c1, c2]
            # Mutate each child in children
            for i, bitstring in enumerate(children):
                bit_list = list(bitstring)
                # For each bit, mutate (flip) it with a probability of 'self.mutation_rate'
                bit_list = [str(int(bit) ^ 1) if random.random() < self.mutation_rate else bit for bit in bit_list]
                bitstring = ''.join(bit_list)
                # Replace the mutated bit
                children[i] = bitstring
            # Calculate
            child_fitness = [self.individual_fitness(child) for child in children]
            for child, child_fitness in zip(children, child_fitness):
                if sum(p != c for p, c in zip(p1, child)) < sum(p != c for p, c in zip(p2, child)):
                    if child_fitness > p1_fitness:
                        self.mating_pool.append(child)
                    else:
                        self.mating_pool.append(p1)
                else:
                    if child_fitness > p2_fitness:
                        self.mating_pool.append(child)
                    else:
                        self.mating_pool.append(p2)

    def plot(self, title=None):
        pass

    def is_finished(self) -> bool:
        pass

    def entropy(self):
        """Calaculates the entropy of a particular generation"""
        bit_count = [0] * self.chromosome_length
        # Count number of 1's at each position in all chromosomes
        for bitstring in self.population:
            for i, bit in enumerate(bitstring):
                if int(bit) == 1:
                    bit_count[i] += 1
        # Find proportions of bits which are 1 in the total population
        p1 = [count / self.population_size for count in bit_count]
        # Find proportions of bits which are zero
        p0 = [1 - pi for pi in p1]
        entropy = 0
        # Calculate entropy as H = -sum(i)[p_i * log2(p_i)]
        for i in range(len(p1)):
            # Assume 0 * log(0) = 0
            if p1[i] != 0:
                entropy -= p1[i] * log(p1[i], 2)
            if p0[i] != 0:
                entropy -= p0[i] * log(p0[i], 2)

        # Store this generation's entropy
        self._entropy.append(entropy)

    def plot_entropy(self):
        pass

    def save_entropy(self):
        pass

    def main(self):
        """Runs the genetic algorithm until either termination criteria is met"""
        self.on_fitness()
        # Run GA loop until max iterations or stopping criteria is reached
        while self.iter_count < self.max_iter and not self.is_finished():
            if self._crowding:
                self.crowding()
            else:
                self.parent_selection()
                self.crossover()
                self.mutation()
            if self._elitism:
                self.elitism()
            self.survivor_selection()
            self.on_fitness()
            self.entropy()
            self.iter_count += 1
            print(f'{self.iter_count}:', max(self.fitness))

            # Plot the desired iterations
            if self.iter_count in self.plot_iter:
                self.plot(title=f'Iteration {self.iter_count}')

        self.plot(title=f'Iteration {self.iter_count}')
        self.save_entropy()
        self.plot_entropy()
