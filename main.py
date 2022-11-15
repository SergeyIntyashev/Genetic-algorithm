import random
import time

import matplotlib.pyplot as plt
import numpy as np
from deap import creator, base, tools
import algelitism

LOW, UP = -5.12, 5.12
ETA = 20
LENGTH_CHROM = 2

ONE_MAX_LENGTH = 100  # длина подлежащей оптимизации битовой строки

POPULATION_SIZE = 200  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума
MAX_GENERATIONS = 50  # максимальное количество поколений
HALL_OF_FAME_SIZE = 5

A = 10

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def random_point(a, b):
    return [random.uniform(a, b), random.uniform(a, b)]


toolbox = base.Toolbox()
toolbox.register('random_point', random_point, LOW, UP)

toolbox.register(
    'individual_creator',
    tools.initIterate,
    creator.Individual,
    toolbox.random_point
)

toolbox.register(
    'population_creator',
    tools.initRepeat,
    list,
    toolbox.individual_creator
)

population = toolbox.population_creator(n=POPULATION_SIZE)


def rastrigin(individual):
    x, y = individual
    f = 2 * A + x ** 2 - A * np.cos(2 * np.pi * x) + y ** 2 - A * np.cos(2 * np.pi * y)
    return f,


def sel_best(population):
    return tools.selBest(population, 1)[0]


toolbox.register("evaluate", rastrigin)
toolbox.register("select", tools.selRoulette)  # селекция по алгоритму Рулетка
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_fit.register('max', np.max)
stats_fit.register('avg', np.mean)

stats_best = tools.Statistics(lambda ind: ind)
stats_best.register('best', sel_best)

mstats = tools.MultiStatistics(fitness=stats_fit, individual=stats_best)


def show(ax, xgrid, ygrid, f):
    ptMax = [[5, -5], [-5, 5], [5, 5], [-5, -5]]

    ax.clear()
    ax.contour(xgrid, ygrid, f)
    ax.scatter(*zip(*ptMax), marker='X', color='red', zorder=1)
    ax.scatter(*zip(*population), color='green', s=2, zorder=0)

    plt.draw()
    plt.gcf().canvas.flush_events()

    time.sleep(0.2)


x = np.arange(LOW, UP, 0.1)
y = np.arange(LOW, UP, 0.1)
xgrid, ygrid = np.meshgrid(x, y)

f_rastrigin = 2 * A + xgrid ** 2 - A * np.cos(2 * np.pi * xgrid) + ygrid ** 2 - A * np.cos(2 * np.pi * ygrid)

plt.ion()

fig, ax = plt.subplots()
fig.set_size_inches(5.12, 5.12)

ax.set_xlim(LOW-3, UP+3)
ax.set_ylim(LOW-3, UP+3)

population, logbook = algelitism.eaSimpleElitism(
    population,
    toolbox,
    cxpb=P_CROSSOVER,
    mutpb=P_MUTATION,
    ngen=MAX_GENERATIONS,
    halloffame=hof,
    stats=mstats,
    callback=(show, (ax, xgrid, ygrid, f_rastrigin)),
    verbose=True
)

max_fitness_values, mean_fitness_values = logbook.chapters['fitness'].select('max', 'avg')

best = hof.items[0]
print(best)

plt.ioff()
plt.show()

plt.plot(max_fitness_values, color='red')
plt.plot(mean_fitness_values, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость макс. и сред. приспособленности от поколения')
plt.show()