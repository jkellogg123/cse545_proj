import os
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
            print(f"{num_completed}/{n} GAs completed")

            ga_result = future.result()
            ga_results.append(ga_result)
            sol = ga_result["best_solution"]
            ga_solutions.append(sol)

            ms = sol.makespan
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

def execute_66(data_sizes: list[int], iterations: int, pop_size: int = 100, num_gens: int = 100, num_ga: int = 24) -> list[dict]:
    """
    Does pretty much all the housekeeping for getting statistics.
    
    Takes a list of custom data sizes to use, *iterations* to average everything over.

    Returned list gives results in same order as *data_sizes*.
    """
    assert iterations >= 1, "Need at least 1 iteration bro."

    ga_params = (pop_size, num_gens)
    all_res = []

    for size in data_sizes:
        print(f"Starting data_size {size}...")
        res = dict()
        Solution.data = create_data(size, seed=69)

        # init res, kinda like a shitty do-while
        print("Iteration 1")
        res = results = run_ga(num_ga, ga_params, Solution.data)
        woc_sol = aggregate(results["ga_solutions"])
        del res["ga_solutions"]
        res["woc_sol"] = woc_sol
        res["woc_ms"] = woc_sol.makespan
        print()

        for i in range(iterations - 1):
            print(f"Iteration {i+2}")
            results = run_ga(num_ga, ga_params, Solution.data)

            check = results["best_sol"]
            if check.makespan < res["best_ms"]:
                res["best_sol"] = check
                res["best_ms"] = check.makespan
            
            woc_sol = aggregate(results["ga_solutions"])
            res["woc_ms"] += woc_sol.makespan
            if woc_sol.makespan < res["woc_sol"].makespan:
                res["woc_sol"] = woc_sol

            add = ("avg_ms", "avg_evolution", "avg_time")
            for stat in add:
                res[stat] += results[stat]

            print()
        
        to_avg = ("avg_ms", "avg_evolution", "avg_time", "woc_ms")
        for stat in to_avg:
            res[stat] /= iterations
        
        all_res.append(res)

    return all_res


# def main():
#     start = time.time()

#     with open("./output/results.txt", 'w') as output_file:
#         def printf(*args, **kwargs):
#             kwargs["file"] = output_file
#             print(*args, **kwargs)
        
#         printf("----------------------------------------\n")

#         data_files = os.listdir("./data")
#         do_files = 2

#         for data_file in data_files[:do_files]:
#             print(f"Processing {data_file}")
#             data_file_id = data_file[3:-4]
#             Solution.data = load_data(data_file)

#             # data_file_id = "RANDOM_4X4"
#             # Solution.data = create_data(4)

#             pop_size = 100
#             num_gens = 100
#             ga_params = (pop_size, num_gens)
#             num_ga = 20

#             results = run_ga(num_ga, ga_params, Solution.data)

#             woc_solution = aggregate(results["ga_solutions"])
#             woc_ms = woc_solution.makespan

#             printf(f"Results from \"{data_file}\"\n")
#             printf(f"Average GA makespan:\t\t {results["avg_ms"]:.2f}")
#             printf(f"Best GA makespan:\t\t\t {results["best_ms"]:.2f}")
#             printf(f"Aggregate makespan:\t\t\t {woc_ms:.2f}")
#             printf(f"Average time per GA:\t\t {results["avg_time"]:.3f}s")
#             printf("\n----------------------------------------\n")

#             xlim = max(results["best_ms"], woc_ms)
#             plot_solution(woc_solution, xlim, "Aggregated", save_path=f"aggregate/{data_file_id}")
#             plot_solution(results["best_sol"], xlim, "Best genetic", save_path=f"ga/{data_file_id}")
#             plot_gens(results["avg_evolution"], f"n={len(Solution.data)}", save_path=data_file_id)

#             print()
        

#         end = time.time()
#         lol = f"\nWhole thing took:\t{end - start:.3f}s"
#         print(lol)
#         printf(lol)


def main():
    start = time.time()

    data_sizes = range(20, 36, 5)
    data_sizes = sorted(list(set(data_sizes)))
    iterations = 30
    num_ga = 24
    print("\033[1;31mExecuting order 66\033[0m\n")
    res_time = time.time()
    results = execute_66(data_sizes, iterations, num_ga=num_ga)
    res_time = time.time() - res_time

    with open("./output/results.txt", 'w') as output_file:
        def printf(*args, **kwargs):
            kwargs["file"] = output_file
            print(*args, **kwargs)
        
        printf("----------------------------------------\n")
        df_data = []

        for size, result in zip(data_sizes, results):
            printf(f"Results for data size {size}\n")
            printf(f"Average GA makespan:\t\t {result["avg_ms"]:.2f}")
            printf(f"Best GA makespan:\t\t\t {result["best_ms"]:.2f}")
            printf(f"Aggregate makespan:\t\t\t {result["woc_ms"]:.2f}")
            printf(f"Average time per GA:\t\t {result["avg_time"]:.3f}s")
            printf("\n----------------------------------------\n")

            xlim = max(result["best_ms"], result["woc_ms"])
            save_name = f"custom_{size}"
            plot_solution(result["woc_sol"], xlim, "Aggregated", save_path=f"aggregate/{save_name}")
            plot_solution(result["best_sol"], xlim, "Best genetic", save_path=f"ga/{save_name}")
            plot_gens(result["avg_evolution"], f"n={size}", save_path=save_name)

            df_data.append((size, result["avg_ms"], result["best_ms"], result["woc_ms"], 
                            (result["avg_ms"] - result["woc_ms"]) / result["avg_ms"] * 100,
                            num_ga * result["avg_time"]))


        columns = ["# of Jobs/Machines", "Avg. Pop. Makespan", "Best Pop. Makespan", "Aggregate Makespan", 
                   "WOC Improvement (%)", 
                   "Pop. Init. Time (s)"]
        df = pd.DataFrame(df_data, columns=columns).round(2)
        df["# of Jobs/Machines"] = df["# of Jobs/Machines"].astype(str)
        df_path = "./output/tables/custom"
        df.to_csv(df_path + ".csv", index=False)
        df.to_html(df_path + ".html", index=False)
        df.to_latex(df_path + ".tex", index=False, float_format="%.2f")
        def df_to_png(df, path):
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 2)
            plt.savefig(path, bbox_inches='tight', dpi=400)
            plt.clf()
        df_to_png(df, df_path + ".png")
        

        end = time.time()
        lol = f"Whole thing took:\t{end - start:.3f}s"
        def printb(*args, **kwargs):
            print(*args, **kwargs)
            printf(*args, **kwargs)
        printb()
        printb(f"Results took:\t{res_time:.3f}s")
        printb(lol)


def reset_output() -> None:
    """
    Clears "./output" directory. Currently does this by deleting it and re-adding it.
    """
    remove = "./output"
    if os.path.exists(remove):
        shutil.rmtree(remove)

    # If I really cared I would make these global variables
    dirs = ("./output/solution_plots/aggregate", "./output/solution_plots/ga", "./output/ga_evolution", "./output/tables")
    for dir in dirs:
        os.makedirs(dir)

if __name__ == "__main__":
    reset_output()
    main()
