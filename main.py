import numpy as np
import os
from solution import Solution, plot_solution
from woc import aggregate
from ga import genetic_algorithm
'''
Main python file. 
Script flow:
1) Process a data file into dataset formatted to solve as an OSSP
2) Run Genetic Algorithm for some number of generations on dataset
    - log GA run data
    - track elite population and best solution
3) Aggregate some "elite" subset of solution from the genetic algorithm 
    to provide a WOC aggregate solution.
    -log WOC aggregation data
4) Ensure WOC solution satisfies OSSP constraints 
5) Compare WOC solution with best GA solution
    - if so, replace best solution
6) Repeat steps 1-6 for all data files.
'''

def load_data(name: str) -> np.ndarray:
    """
    Loads data structure from given file name under "./data" directory.
    """
    path = "./data/" + name
    with open(path, 'r') as file:
        file.readline()
        line = file.readline()
        n = int(line.strip().split(' ')[0])

        data = []
        file.readline()
        for i in range(n):
            data.append(file.readline().strip().split(' '))
        data = np.array(data, dtype=np.int16)
        
        # the way the data is formatted is goofy, this "corrects" it
        file.readline()
        sort = []
        for i in range(n):
            sort = np.array(file.readline().strip().split(' '), dtype=np.int8) - 1
            data[i][sort] = np.copy(data[i])    # copy is needed because it overwrites element by element, unlike tuple assignment for example
        
    return data.T

def create_data(n: int, max_time=100, seed: int = None) -> np.ndarray:
    """
    Creates a random dataset with *n* machines and *n* jobs, where each activity takes no more than *max_time* time units exclusive.
    """
    np.random.seed(seed)
    return np.random.randint(1, max_time, size=(n, n))


def main():
    file = "tai1515_0.txt"
    Solution.data = load_data(file)
    best_ms = float('inf')
    ga_solutions = []
    for _ in range(20):
        ga_solution = genetic_algorithm(100, 100)
        ga_solutions.append(ga_solution)
        ms = ga_solution.calc_makespan()
        if  ms < best_ms:
            best_ms = ms

    woc_solution = aggregate(ga_solutions)
    
    woc_ms = woc_solution.calc_makespan()
    print("Results:")
    print("Aggregate solution:\n",woc_solution.schedule)
    print("Aggregate makespan",woc_ms)
    print("Best makespan for single GA",best_ms)
    print("Mean makespan from GA",np.mean([x.makespan for x in ga_solutions]))
    plot_solution(woc_solution)

if __name__ == "__main__":
    main()
