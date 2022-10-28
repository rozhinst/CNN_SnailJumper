from player import Player
import numpy as np
from config import CONFIG
import random
import copy
import pandas as pd

MUTATION_PROB = 0.2
NORMAL_MEAN = 0
NORMAL_STD = 0.5


class Evolution():

    def __init__(self, mode):
        self.mode = mode
        self.fitness_max = []
        self.fitness_avg = []
        self.fitness_min = []

    # calculate fitness of players

    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):
        global MUTATION_PROB
        # MUTATION_PROB += 0.00001
        # print(MUTATION_PROB)

        for i, x in enumerate(child.nn.biases):
            if(random.random() < MUTATION_PROB):
                child.nn.biases[i] += np.random.normal(
                    NORMAL_MEAN, NORMAL_STD, size=np.shape(x))

        for i, x in enumerate(child.nn.weights):
            if(random.random() < MUTATION_PROB):
                child.nn.weights[i] += np.random.normal(
                    NORMAL_MEAN, NORMAL_STD, size=np.shape(x))

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            new_players = []
            # No Crossover
            for player in prev_players:
                new_player = copy.deepcopy(player)
                self.mutate(new_player)
                new_players.append(new_player)
            # Crossover
            # for i in range(0, len(prev_players), 2):
            #     parentA = prev_players[i]
            #     parentB = prev_players[i + 1]
            #     # First Child
            #     new_playerA = copy.deepcopy(parentA)
            #     new_playerA.nn.weights = copy.deepcopy(parentB.nn.weights)
            #     self.mutate(new_playerA)
            #     # Second Child
            #     new_playerB = copy.deepcopy(parentB)
            #     new_playerB.nn.weights = copy.deepcopy(parentA.nn.weights)
            #     self.mutate(new_playerB)
            #     new_players.extend([new_playerA, new_playerB])
            return new_players

    def next_population_selection(self, players, num_players):

        self.save_fitness_stat(players)
        # Best Fitness
        new_players = sorted(players, key=lambda x: x.fitness, reverse=True)
        return new_players[:num_players]
        # Weighted Random
        # return random.choices(players, weights=[x.fitness for x in players], k=num_players)

    def save_fitness_stat(self, players):
        fitness_arr = [x.fitness for x in players]
        self.fitness_max.append(max(fitness_arr))
        self.fitness_avg.append(sum(fitness_arr) / len(fitness_arr))
        self.fitness_min.append(min(fitness_arr))

        df = pd.DataFrame(
            {'max': self.fitness_max, 'min': self.fitness_min, 'avg': self.fitness_avg})

        df.to_csv('a.csv')
