import numpy as np
import os
from solution import Solution


def load_data(name):
    path = "./data/" + name
    with open(path, 'r') as file:
        file.readline()
        line = file.readline()
        n = int(line.strip().split(' ')[0])

        data = []
        file.readline()
        for i in range(n):
            data.append(file.readline().strip().split(' '))
        data = np.array(data, dtype=np.uint8)
        
        # the way the data is formatted is goofy, this "corrects" it
        file.readline()
        sort = []
        for i in range(n):
            sort = np.array(file.readline().strip().split(' '), dtype=np.uint8)
            data[i] = data[i][sort - 1]
        
        return data


def main():
    # file = "tai44_0.txt"
    # data = Solution.data = load_data(file)

    for file in os.listdir("data"):
        print(file)
        print(load_data(file))
        print()
    print("I am richard")

if __name__ == "__main__":
    main()