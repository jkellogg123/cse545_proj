import numpy as np
from collections.abc import Iterable
from solution import Solution

# a set of orders in which machines execute tasks. 
# each numerical value is the job a task corresponds to. 
test_set = [[[2,3,1],[1,3,2],[3,2,1]],
            [[2,3,1],[1,3,2],[3,2,1]],
            [[1,3,2],[2,3,1],[3,2,1]],
            [[2,3,1],[1,3,2],[3,2,1]],
            [[2,3,1],[1,3,2],[3,2,1]],
            [[1,3,2],[2,3,1],[3,2,1]]]
class Woc:
    weights = None
    def __init__(self, experts: np.ndarray | list) -> None:
        '''
        Summary:
        This class creates an agreement matrix for  of dimensions (N, N, M),
        where N is the number of jobs an M is the number of machines.

        Output:
        The agreement matrix will aggregate into an unresolved solution
        i.e. the gaps between tasks resulting from delays in jobs running on 
        multiple machines at once have not been resolved

        Example:
        Given an expert population, size three, where
        solution 1:
        [[2,3,1],
         [1,2,3],
         [3,2,1]]
        solution 2:
        [[2,3,1],
         [1,2,3],
         [3,2,1]]
        solution 3:
        [[1,3,2],
         [2,1,3],
         [3,2,1]]

        The approximate resulting A:
        [[[0.33, 0.66, 0.00], [0.00, 0.00, 1.00], [0.66, 0.33, 0.00]],
          [0.66, 0.33, 0.00], [0.00, 0.66, 0.33], [0.00, 0.00, 1.00]],
          [0.00, 0.00, 1.00], [0.00, 1.00, 0.00], [1.00, 0.00, 0.00]]]
        Aggregated solution:
        [[2,3,1],
         [1,2,3],
         [3,2,1]]
        '''
        self.experts = experts

        self.P, self.M, self.N = np.array(experts).shape

        self.A = np.zeros([self.M, self.N, self.N])
    
    def print_A(self):
        print(self.A)

    def find_agreement(self) -> None:
        '''
        agreement based on job_task sequence for each schedule from the population.
        '''
        for p in range(self.P):
            for m in range(self.M):
                for n in range(self.N):
                    job_task = self.experts[p][m][n]
                    self.A[m][n][job_task]+=self.weights[p]

    def create_solution(self) -> np.ndarray:
        solution = np.full((self.M, self.N), -1)
        flattened_ranks = np.argsort(self.A, axis=None)
        # print(flattened_ranks)
        machine, task, job_task = np.unravel_index(np.flip(flattened_ranks), self.A.shape)
        # print(self.A.shape)
        # print(self.A)
        # print(f"{machine} \n{task} \n{job_task}")
        for x in range(len(machine)):
            if -1 not in solution:
                break
            m = machine[x]
            n1 = task[x]
            n2 = job_task[x]
            # print(solution)
            # print(f"m: {m} | n1: {n1} | n2: {n2}")
            # input()
            if solution[m][n1] == -1:
                solution[m][n1] = n2

        # for m in range(self.M):
        #     for n in range(self.N):
        #         for i in indices:
        #             if not i in solution[m]:
        #                 solution[m][n] = i
        #                 break
        return solution

        

def aggregate(sols: Iterable[Solution]) -> Solution:
    """
    Takes a collection of *Solution* objects and aggregates them into one (hopefully better) solution.
    """
    scheds = [sol.schedule for sol in sols]
    woc = Woc(scheds)
    woc.weights = [sol.makespan for sol in sols]
    total_weight = np.sum(woc.weights)
    for weight in woc.weights: weight/=total_weight
    woc.find_agreement()
    return Solution(woc.create_solution())
