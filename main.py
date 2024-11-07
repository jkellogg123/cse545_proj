import numpy as np
import os
from solution import Solution, plot_solution
from woc import aggregate
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

def load_data(name):
    path = "./data/" + name
    with open(path, 'r') as file:
        file.readline()
        line = file.readline()
        n = int(line.strip().split(' ')[0])

        data = []
        file.readline()
        for i in range(n):
            data.append(file.readline().strip().split(' '))
        data = np.array(data, dtype=np.uint8)
        
        # the way the data is formatted is goofy, this "corrects" it
        file.readline()
        sort = []
        for i in range(n):
            sort = np.array(file.readline().strip().split(' '), dtype=np.uint8) - 1
            data[i][sort] = np.copy(data[i])    # copy is needed because it overwrites element by element, unlike tuple assignment for example
        
    return data.T


def main():
    file = "tai44_0.txt"
    Solution.data = load_data(file)
    # sol = Solution()
    
    sols = [Solution() for _ in range(50)]
    sol = aggregate(sols)

    print(sol.data)
    print(sol.schedule)
    print(sol.starts)
    print()
    print(sol.calc_makespan())
    plot_solution(sol)

    # Iterate through each data file
    # for file in os.listdir("data"):
    #     print(file)
    #     Solution.data = load_data(file)
    #     print(Solution.data)

if __name__ == "__main__":
    main()
