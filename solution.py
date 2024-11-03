from __future__ import annotations
import numpy as np

class Solution:
    """
    Object to represent a single solution.

    Attributes:
        schedule (2D np array):         schedule[i, j] is the jth job of the ith machine
        starts (2D np array):           starts[i, j] is the starting time of the jth job of ith machine

        data (static, 2D np array):     data[i, j] is the time-length of the jth job on the ith machine
        cross_rate (static, float):     0 <= rate <= 1, denotes frequency of crossover operation
        mutate_rate (static, float):    0 <= rate <= 1, denotes frequency of mutate operation

    """

    data = None
    cross_rate = 0.75
    mutate_rate = 0.02

    def __init__(self, n: int, rand=True):
        self.schedule = np.full((n, n), -1)
        self.starts = np.full((n, n), -1)
        if rand and not self.data is None:
            # TODO
            # Create random solution for initializing population of chromosomes for genetic algorithm
            # Maybe doesn't need to be random, but need some way to populate our initial generation
            pass
    
    def job_times(self) -> np.ndarray:
        """
        Returns array of finishing times for each machine.
        """

        if np.isin(-1, self.schedule) or np.isin(-1, self.starts) or self.data is None:
            return None
        
        res = np.empty(self.schedule.shape[0])
        for i in range(len(res)):
            res[i] = self.starts[i, -1] + self.data[i, self.schedule[i, -1]]
        
        return res

    def makespan(self) -> float:
        """
        Returns the makespan of the solution (total finishing time).
        """
        jt = self.job_times()
        if jt is None:
            return -1
        else:
            return np.max(jt)

    def mutate(self):
        pass

    def crossover(s1, s2: Solution) -> tuple[Solution, Solution]:
        pass
