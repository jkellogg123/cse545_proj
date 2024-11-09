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

        self.A = np.zeros([self.N, self.M, self.M])
        self.A_shape = np.shape(self.A)
    
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
        for m in range(self.M):
            
            for n in range(self.N):
                indices = np.argsort(self.A[m][n])
                for i in indices:
                    if not i in solution[m]:
                        solution[m][n] = i
                        break
        return solution

    def solution(self):
        self.find_agreement()
        return self.create_solution()
        

def aggregate(sols: Iterable[Solution]) -> Solution:
    """
    Takes a collection of *Solution* objects and aggregates them into one (hopefully better) solution.
    """
    scheds = [sol.schedule for sol in sols]
    woc = Woc(scheds)
    woc.weights = [sol.makespan for sol in sols]
    woc.find_agreement()
    return Solution(woc.create_solution())

def aggregate_sequence (solutions: list[Solution]) -> Solution:
    '''
    aggregate solution based on sequence of tasks for each machine

    Parameters - solutions: list of experts to be aggregated.

    Returns - new solution object that represents aggregated schedule of tasks
    '''

    num_machines, num_jobs = solutions[0].schedule.shape
    agreement_matrice = np.zeros((num_machines, num_jobs, num_jobs), dtype = int)

    for sol in solutions: #populate agreement , should loop through experts and count job->job transitions
        for machine in range(num_machines):
            for j in range(num_jobs - 1):
                current_job = sol.schedule[machine, j]
                next_job = sol.schedule[machine, j + 1] 
                agreement_matrice[machine, current_job, next_job] += 1 

    consensus_schedule = np.zeros((num_machines, num_jobs) ,dtype = int) # get the consensus sequence which is based off of max agreement

    for machine in range(num_machines):  # loop to determine most agreed-upon sequnce of jobs for each machine
        current_job = np.argmax(agreement_matrice[machine].sum (axis = 1))
        visited = {current_job}
        consensus_schedule[machine, 0] = current_job

        '''
        Determining next job based off max agreement
        '''
        for j in range(1, num_jobs):
            next_job = np.argmax(agreement_matrice[machine, current_job])
            while next_job in visited:
                agreement_matrice[machine, current_job, next_job] = -1 # this marks task used
                next_job = np.argmax(agreement_matrice[machine, current_job])
            consensus_schedule[machine, j] = next_job
            visited.add(next_job)
            current_job = next_job
    
    return Solution(schedule = consensus_schedule)
