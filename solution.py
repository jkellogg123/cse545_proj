from __future__ import annotations
import numpy as np
import bisect

class Solution:
    """
    Object to represent a single solution.

    Attributes:
        schedule (2D np array):         schedule[i, j] is the jth job of the ith machine
        starts (2D np array):           starts[i, j] is the starting time of the jth job of ith machine
        makespan (float):               total time of solution. Initially -1 until calculated with calc_makespan()

        data (static, 2D np array):     data[i, j] is the time-length of the jth job on the ith machine
        cross_rate (static, float):     0 <= rate <= 1, denotes frequency of crossover operation
        mutate_rate (static, float):    0 <= rate <= 1, denotes frequency of mutate operation

    """

    data = None
    task_jobs = None
    cross_rate = 0.75
    mutate_rate = 0.02

    def __init__(self, schedule: np.ndarray=None):
        """
        If *schedule* is given, associates a valid solution with start times

        Otherwise, creates a random solution
        """
        assert not self.data is None, "Initialize Solution.data before instantiating Solution objects"

        self.makespan = -1
        if schedule is None:
            # Create random solution
            self.schedule = random_schedule(self.data.shape)
            self.starts = make_starts(self.schedule)
        else:
            # Assign starts to given schedule
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
    """
    assert not Solution.data is None, "Need to initialize Solution.data before make_starts can run."
    if schedule is None:
        return None

    shape = schedule.shape
    starts = np.empty(shape)
    jobs_busy = [[] for _ in range(shape[1])]     # jobs_busy[i] gives list of (start, end) times where the ith job is busy
    for col in range(shape[1]):
        for row in range(shape[0]):
            if col != 0:
                prev_job_ind = (row, col-1)
                prev_job = schedule[prev_job_ind]
                after = starts[prev_job_ind] + Solution.data[row, prev_job]
            else:
                after = 0
            job = schedule[row, col]
            length = Solution.data[row, job]
            start = insert_job(after, jobs_busy[job], length)
            starts[row, col] = start
            bisect.insort(jobs_busy[job], (start, start + length), key=lambda x: x[0])

    return starts

def insert_job(after: float, job_starts: list, length: float) -> float:
    """
    Returns the best starting time for a job that takes *length* time, inserted after *after*, with the given list *job_starts* of busyness for the job

    Used in *make_starts*
    """
    if not job_starts:
        return after
    
    # Fits in any gap?
    for i in range(len(job_starts)):
        start, end = job_starts[i]
        # Too early
        if end < after:
            continue

        if i == 0:
            prev_end = -1
        else:
            prev_end = job_starts[i-1][1]
        insert_check = max(after, prev_end)

        # Fits before?
        if start - length >= insert_check:
            return insert_check
        else:
            continue
    
    # Doesn't fit in gaps
    return max(after, job_starts[-1][1])
