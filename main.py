import numpy as np
from solution import Solution



def load_data(path):
    pass

def main():
    n = 10
    path = "69"

    Solution.data = load_data(path)
    lol = Solution(n)
    print(lol.data)


if __name__ == "__main__":
    main()