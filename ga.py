import random
import numpy as np
import time
import matplotlib.pyplot as plt
from solution import Solution

def mutate(sol: Solution) -> Solution:
    """
    Returns mutated version of *sol*.
    """
    if random.random() < Solution.mutate_rate:
        machine = random.randint(0, sol.schedule.shape[0] - 1)
        job1, job2 = random.sample(range(sol.schedule.shape[1]), 2)
        new_schedule = np.copy(sol.schedule)
        new_schedule[machine, job1], new_schedule[machine, job2] = sol.schedule[machine, job2], sol.schedule[machine, job1]
        res = Solution(new_schedule)
    else:
        res = sol

    return res

def crossover(s1: Solution, s2: Solution) -> tuple[Solution, Solution]:
    """
    Returns offspring pair of *s1* and *s2*.
    """
    
    if random.random() < Solution.cross_rate:
        # Want unique objects
        if id(s1) == id(s2):
            s2 = Solution(s2.schedule)
        num_machines, num_jobs = s1.schedule.shape
        new_schedule1 = np.empty_like(s1.schedule)
        new_schedule2 = np.empty_like(s2.schedule)
        
        for machine in range(num_machines):
            start, end = sorted(random.sample(range(num_jobs), 2))
            
            # Copy the selected slice from each parent to the corresponding child
            new_schedule1[machine, start:end] = s1.schedule[machine, start:end]
            new_schedule2[machine, start:end] = s2.schedule[machine, start:end]
            
            # Fill remaining positions from the other parent, avoiding duplicates
            fill_from_parent(new_schedule1[machine], s2.schedule[machine], start, end)
            fill_from_parent(new_schedule2[machine], s1.schedule[machine], start, end)
        
        # Chooses best two from children and parent solutions
        c1, c2 = Solution(new_schedule1), Solution(new_schedule2)
        return sorted([c1, c2, s1, s2], key=lambda x: x.calc_makespan())[:2]
    
    return s1, s2

def fill_from_parent(child_row, parent_row, start, end):
   
    pos = end  # Start filling after the crossover slice
    for job in parent_row:
        if job not in child_row[start:end]:
            if pos >= len(child_row):
                pos = 0  # Wrap around if we reach the end of the row
            child_row[pos] = job
            pos += 1

def genetic_algorithm(population_size: int, generations: int) -> dict:
    """
    Runs ga with given parameters and returns dictionary of results.
    """
    start = time.process_time()
    
    population = [Solution() for _ in range(population_size)]
    evolution = []
    
    for gen in range(generations):
        for sol in population:
            sol.calc_makespan()

        population.sort(key=lambda x: x.makespan)
        evolution.append(population[0].makespan)
        
        next_gen = population[:population_size // 2]

        while len(next_gen) < population_size:
            parent1, parent2 = random.sample(next_gen, 2)
            offspring1, offspring2 = crossover(parent1, parent2)
            next_gen.append(mutate(offspring1))
            if len(next_gen) < population_size:
                next_gen.append(mutate(offspring2))

        population = next_gen

    best_solution = min(population, key=lambda x: x.makespan)

    end = time.process_time()
    results = {"best_solution" : best_solution,
               "evolution" : evolution,
               "time" : end - start,}
    return results


def plot_gens(gens: list[float], title: str = None, save_path : str = None) -> None:
    """
    Takes list of makespan per generation and plots its evolution.

    If *save_path* is given as a nonempty string, saves plot at "./output/ga_evolution/" under *save_path*.png
    """

    plt.plot(range(1, len(gens)+1), gens)

    plt.title("Genetic Algorithm Evolution" + f" ({title})" if title else "")
    plt.xlabel("Generation #")
    plt.ylabel("Makespan")
    if save_path:
        path = f"./output/ga_evolution/{save_path}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()