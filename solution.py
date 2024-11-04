from __future__ import annotations
import numpy as np
import random

class Solution:
    """
    Object to represent a single solution.

    Attributes:
        schedule (2D np array):         schedule[i, j] is the jth job of the ith machine
        starts (2D np array):           starts[i, j] is the starting time of the jth job of ith machine
        makespan (float):               the total time of the solution. If it hasn't been calculated yet, initial value is -1

        data (static, 2D np array):     data[i, j] is the time-length of the jth job on the ith machine
        cross_rate (static, float):     0 <= rate <= 1, denotes frequency of crossover operation
        mutate_rate (static, float):    0 <= rate <= 1, denotes frequency of mutate operation

    """

    data = None
    cross_rate = 0.75
    mutate_rate = 0.02

    def __init__(self, schedule: np.ndarray=None):
        """
        If schedule is given, associates a valid solution with start times

        Otherwise, creates a random solution
        """
        assert not self.data is None, "Initialize data before instantiating Solution objects"

        self.makespan = -1

        if schedule is None:
            # Create random solution
            shape = self.data.shape
            self.schedule = random_schedule(shape)
            self.starts = make_starts(self.schedule)
        else:
            # Create solution with given schedule, associate a valid starts array
            self.schedule = schedule
            self.starts = make_starts(schedule)
    
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

    def calc_makespan(self) -> float:
        """
        Returns and sets the makespan of the solution (total finishing time).
        """
        jt = self.job_times()
        if jt is None:
            return -1
        else:
            ms = np.max(jt)
            self.makespan = ms
            return ms


def random_schedule(shape: tuple[int, int]) -> np.ndarray:
    num_machines, num_jobs = shape
    return np.array([np.random.permutation(num_jobs) for _ in range(num_machines)])


def make_starts(schedule: np.ndarray) -> np.ndarray:
    """
    Returns a valid starts array corresponding to given schedule

    *NOTE: Currently just returns an array of -1s lol*
    """
    if schedule is None:
        return None

    shape = schedule.shape
    starts = np.full(shape, -1)
    

    return starts