import numpy as np
import os
from solution import Solution
from woc import Woc
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
5) Compare WOC solution with best GA solutiong
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
        # didn't correctly sort the data... I changed it to sort times by corresponding job.... 
        # but I don't think this addresses the problem at hand. 
        # I propose we use a list of hash maps/ dictionaries or c like struct.
        # Example data:
        # data[0] = {start_time: 0, runtime: 64, job_id: 5}
        # It's a bit of memory/compute overhead but this seems easier to digest and come up with solutions...
        file.readline()
        sort = []
        for i in range(n):
            sort = np.array(file.readline().strip().split(' '), dtype=np.uint8) - 1
            data[i][sort] = np.copy(data[i])    # copy is needed because it overwrites element by element, unlike tuple assignment for example
        
    return data.T


def main():
    file = "tai44_0.txt"
    Solution.data = load_data(file)
    sol = Solution()
    print(sol.data)
    print(sol.schedule)
    print(sol.starts)
    print()
    print(sol.calc_makespan())
    # woc_solver = Woc(solution_array)
    # solution = woc_solver.solution() #aggregate schedule
    

    # for file in os.listdir("data"):
    #     print(file)
    #     Solution.data = load_data(file)
        
    #     ga_solver = Solution()
    #     for d in ga_solver.data:
    #         print(d)
    #     print(Solution.data)

if __name__ == "__main__":
    main()