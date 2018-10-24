import operator
import random

import numpy

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class PSO:

    toolbox = None

    def generate(self, size, pmin, pmax, smin, smax):
        part = creator.Particle(random.uniform(pmin, pmax)
                                for _ in range(size))

        part.speed = [random.uniform(smin, smax) for _ in range(size)]

        part.smin = smin
        part.smax = smax
        return part

    def update_particle(self, part, best, c1, c2):
        u1 = (random.uniform(0, c1) for _ in range(len(part)))
        u2 = (random.uniform(0, c2) for _ in range(len(part)))

        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))

        part.speed = map(operator.add, part.speed,
                         map(operator.add, v_u1, v_u2))

        inertia_weights = (random.uniform(0.4, 0.9) for _ in range(len(part)))

        part.speed = list(map(operator.mul, inertia_weights, part.speed))

        for i, speed in enumerate(part.speed):
            if speed < part.smin:
                part.speed[i] = part.smin
            elif speed > part.smax:
                part.speed[i] = part.smax

        part[:] = list(map(operator.add, part, part.speed))

    def process(self):
        pop = self.toolbox.population(n=200)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        GEN = 300
        best = None

        all_bests = []

        for g in range(GEN):
            for part in pop:
                part.fitness.values = self.toolbox.evaluate(part)

                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values

                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values

            all_bests.append(best.fitness.values)

            for part in pop:
                self.toolbox.update(part, best)

            # Gather all the fitnesses in one list and print the stats
            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            # print(logbook.stream)

        return pop, logbook, best, all_bests

    def plot(self, log):

        x_values = []
        y_values = []

        for i in range(len(log)):
            y_values.append(log[i]['avg'])
            x_values.append(log[i]['gen'])

        plt.plot(x_values, y_values, 'k')
        plt.yscale('log')
        plt.ylabel('Avg')
        plt.xlabel('Generation')
        plt.show()

    def __init__(self):

        #Create Fitness and Individual
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
        creator.create("Particle", list, fitness=creator.Fitness,
                    speed=list, smin=None, smax=None, best=None)

        self.toolbox = base.Toolbox()
        
        pmin = pow(2, -15)
        pmax = pow(2, 3)

        #Set Individual values
        self.toolbox.register("particle", self.generate, size=10, pmin=pmin,
                        pmax=pmax, smin=pmin*0.1, smax=pmax*0.1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)

        self.toolbox.register("update", self.update_particle, c1=2.0, c2=2.0)
        self.toolbox.register("evaluate", benchmarks.h1)

    def main(self, plot):
        results = self.process()
        if (plot):
            print('print sem plot')
            self.plot(results[1])
        return results
