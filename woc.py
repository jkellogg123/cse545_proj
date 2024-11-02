import numpy as np
import scipy.special as sc
import math
import sys

test_set = [[[2,3,1],[1,3,2],[3,2,1]],
            [[2,3,1],[1,3,2],[3,2,1]],
            [[1,3,2],[2,3,1],[3,2,1]],
            [[2,3,1],[1,3,2],[3,2,1]],
            [[2,3,1],[1,3,2],[3,2,1]],
            [[1,3,2],[2,3,1],[3,2,1]]]
class Woc:
    def __init__(self, experts: np.ndarray | list, b1:float=0, b2:float=0) -> None:
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
        self.b1 = b1 # beta value 1 
        self.b2 = b2
        self.A = np.zeros([self.N, self.M, self.M])
        self.A_shape = np.shape(self.A)
    
    def print_A(self):
        print(self.A)

    def find_agreement(self) -> None:
        for p in range(self.P):
            for m in range(self.M):
                for n in range(self.N):
                    job_task = self.experts[p][m][n]-1
                    self.A[m][n][job_task]+=1/self.P
        # print(self.A)

    def create_solution(self) -> np.ndarray:
        solution = np.zeros((self.M, self.N))
        for m in range(self.M):
            for n in range(self.N):
                solution[m][n] = int(np.argmax(self.A[m][n]))+1

        # print(solution)

    def solution(self):
        self.find_agreement()
        self.create_solution()
        