import random
import numpy as np
from solution import Solution 

def mutate(sol: Solution) -> Solution:
   
    if random.random() < Solution.mutate_rate:
        machine = random.randint(0, sol.schedule.shape[0] - 1)
        job1, job2 = random.sample(range(sol.schedule.shape[1]), 2)
        sol.schedule[machine, job1], sol.schedule[machine, job2] = sol.schedule[machine, job2], sol.schedule[machine, job1]
        sol.starts = make_starts(sol.schedule)
    return sol


def crossover(s1: Solution, s2: Solution) -> tuple[Solution, Solution]:
   #Create list of random solutions 
   #initial chromasomeons solutions in range of 100 [Solution() for_in range(whatever number)]
    if random.random() < Solution.cross_rate:
        crossover_point = random.randint(1, s1.schedule.shape[1] - 1)
        new_schedule1 = np.copy(s1.schedule)
        new_schedule2 = np.copy(s2.schedule)
        for machine in range(s1.schedule.shape[0]):
            new_schedule1[machine, crossover_point:], new_schedule2[machine, crossover_point:] = (
                s2.schedule[machine, crossover_point:],
                s1.schedule[machine, crossover_point:]
            )
        return Solution(new_schedule1), Solution(new_schedule2)
    return s1, s2


def genetic_algorithm(population_size: int, generations: int) -> Solution:
   
   
    population = [Solution() for _ in range(population_size)]
    
    for gen in range(generations):
        for sol in population:
            sol.calc_makespan()

        population.sort(key=lambda x: x.makespan)
        
        next_gen = population[:population_size // 2]

        while len(next_gen) < population_size:
            parent1, parent2 = random.sample(next_gen, 2)
            offspring1, offspring2 = crossover(parent1, parent2)
            next_gen.append(mutate(offspring1))
            if len(next_gen) < population_size:
                next_gen.append(mutate(offspring2))

        population = next_gen

    best_solution = min(population, key=lambda x: x.makespan)
    return best_solution