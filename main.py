import os
import shutil
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from solution import Solution, plot_solution
from woc import aggregate
from ga import genetic_algorithm, plot_gens
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

# Need this wrapper because processes have unique address spaces, so Solution.data doesn't hold.
def ga_process(ga_params: tuple, data: np.ndarray) -> dict:
    """
    Process function that runs ga with given *ga_params*.

    *data* is needed to reset address space.
    """
    Solution.data = data
    return genetic_algorithm(*ga_params)


def run_ga(n: int, ga_params: tuple, data: np.ndarray) -> dict:
    """
    Runs the genetic algorithm *n* times and returns a dict of (mostly statistic) results.
    """
    cores = max(1, (2 * os.cpu_count()) // 3)       # Dedicate 2/3 of logical processors to producing genetic results
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = [executor.submit(ga_process, ga_params, data) for _ in range(n)]
        ga_results = []
        ga_solutions = []
        best_ms = float('inf')

        for num_completed, future in enumerate(as_completed(futures), 1):
            print(f"{num_completed} GAs completed")

            ga_result = future.result()
            ga_results.append(ga_result)
            sol = ga_result["best_solution"]
            ga_solutions.append(sol)

            ms = sol.calc_makespan()
            if  ms < best_ms:
                best_ms = ms
                best_sol = sol

    results = {"ga_solutions" : ga_solutions,
               "best_sol" : best_sol,
               "best_ms" : best_ms,
               "avg_ms" : np.mean([sol.makespan for sol in ga_solutions]),
               "avg_evolution" : np.mean([result["evolution"] for result in ga_results], axis=0),
               "avg_time" : np.mean([result["time"] for result in ga_results])}
    return results

def main():
    start = time.time()

    with open("./output/results.txt", 'w') as output_file:
        def printf(*args, **kwargs):
            kwargs["file"] = output_file
            print(*args, **kwargs)
        
        printf("----------------------------------------\n")

        data_files = os.listdir("./data")
        do_files = 2

        for data_file in data_files[:do_files]:
            print(f"Processing {data_file}")
            data_file_id = data_file[3:-4]
            Solution.data = load_data(data_file)

            pop_size = 100
            num_gens = 100
            ga_params = (pop_size, num_gens)
            num_ga = 8

            results = run_ga(num_ga, ga_params, Solution.data)

            woc_solution = aggregate(results["ga_solutions"])
            woc_ms = woc_solution.calc_makespan()

            printf(f"Results from \"{data_file}\"\n")
            printf(f"Average GA makespan:\t\t {results["avg_ms"]:.2f}")
            printf(f"Best GA makespan:\t\t\t {results["best_ms"]:.2f}")
            printf(f"Aggregate makespan:\t\t\t {woc_ms:.2f}")
            printf(f"Average time per GA:\t\t {results["avg_time"]:.3f}s")
            printf("\n----------------------------------------\n")

            xlim = max(results["best_ms"], woc_ms)
            plot_solution(woc_solution, xlim, "Aggregated", save_path=f"aggregate/{data_file_id}")
            plot_solution(results["best_sol"], xlim, "Best genetic", save_path=f"ga/{data_file_id}")
            plot_gens(results["avg_evolution"], f"n={len(Solution.data)}", save_path=data_file_id)

            print()
        

        end = time.time()
        lol = f"\nWhole thing took:\t{end - start:.3f}s"
        print(lol)
        printf(lol)


def reset_output() -> None:
    """
    Clears "./output" directory. Currently does this by deleting it and readding it.
    """
    remove = "./output"
    if os.path.exists(remove):
        shutil.rmtree(remove)

    # If I really cared I would make these global variables
    dirs = ("./output/solution_plots/aggregate", "./output/solution_plots/ga", "./output/ga_evolution")
    for dir in dirs:
        os.makedirs(dir)

if __name__ == "__main__":
    reset_output()
    main()
