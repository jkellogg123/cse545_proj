import numpy as np

# a set of orders in which machines execute tasks. 
# each numerical value is the job a task corresponds to. 
test_set = [[[2,3,1],[1,3,2],[3,2,1]],
            [[2,3,1],[1,3,2],[3,2,1]],
            [[1,3,2],[2,3,1],[3,2,1]],
            [[2,3,1],[1,3,2],[3,2,1]],
            [[2,3,1],[1,3,2],[3,2,1]],
            [[1,3,2],[2,3,1],[3,2,1]]]
class Woc:
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
                    self.A[m][n][job_task]+=1/self.P

    def create_solution(self) -> np.ndarray:
        solution = np.zeros((self.M, self.N))
        for m in range(self.M):
            for n in range(self.N):
                solution[m][n] = int(np.argmax(self.A[m][n]))
        return solution

    def solution(self):
        self.find_agreement()
        return self.create_solution()
        