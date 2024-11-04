from __future__ import annotations
import numpy as np
import random
import copy

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

    def __init__(self, n: int, raw_data: np.ndarray, rand=True):
        self.schedule = np.full((n, n), -1)
        self.starts = np.full((n, n), -1)
        self.raw_data = raw_data
        self.data = self.random_machines()
        self.raw_times()


    def random_machines (self) -> dict:
        '''
        just an example of the structured data... incomplete of course.
        I do think each of these steps is going to get to be a lot of code...
        but maybe we can solve it elegantly
        '''
        data = []
        for row in self.raw_data:
            machine = []
            for i in range(len(row)):
                machine.append({"job": i+1, "start": 0, "run": row[i]})
            random.shuffle(machine)
            
            data.append(machine)
        return data

    def raw_times(self):
        for machine in self.data:
            for i in range(1, len(machine)):
                machine[i]["start"] = machine[i-1]["start"] + machine[i-1]["run"]+1

    def add_waits(self):
        pass

    def mutate(self):
        pass

    def crossover(s1, s2: Solution) -> tuple[Solution, Solution]:
        pass
