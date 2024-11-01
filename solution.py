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

    def mutate(self):
        pass

    def crossover(s1, s2: Solution) -> tuple[Solution, Solution]:
        pass
