# Author: Liam Maher
# UserID: lkm0049@auburn.edu
# Version: 03.23.2024
# Class: COMP 3270
# Assignment: Programming Assignment 1
# Description: Implement a program that will read a file with integers and display different algorithms for Matrix Chain Multiplication.


#Importing necessary libraries from directions
#MAKE SURE TO CHECK IF TIME IMPORT IS ACTUALLY NEEDED
import pandas as pd
import numpy as np
import time
import pandas as pd
import time


class MatrixMultiplication:

    #Constructor for the class
    def __init__(self, file_name):
        self.file_name = file_name
        self.input_matrices = None

    #Method to read input file and create the matrices
    #CODE GENERATED FROM COPILOT, CHECK TO MAKE SURE CORRECT
    def read_input_file(self):
        with open(self.file_name, 'r') as file:
            lines = file.readlines()
            sequences = lines[0].strip().split(";")
            sequences = [list(map(int, sequence.strip().split(","))) for sequence in sequences]
            self.input_matrices = [np.array(sequence).reshape(4, 4) for sequence in sequences]
    
    #Method to complete the creation of the input matrices with the given integer values
    #CODE GENERATED FROM COPILOT, CHECK TO MAKE SURE CORRECT
    def create_input_matrices(self):
        matrices = []
        for i in range(0, len(self.input_matrices), 2):
            if self.input_matrices[i].shape[0] > 0:
                matrices.append((self.input_matrices[i][0], self.input_matrices[i][1]))
        return matrices
    
    #Method to measure the time it takes to mutiply the matrices
    #CODE GENERATED FROM COPILOT, CHECK TO MAKE SURE CORRECT
    def measure_time(self, algorithm, matrices):
        start_time = time.time()

        if algorithm == 1:
            MatrixMultiplication.algorithm_1(*matrices)
        elif algorithm == 2:
            MatrixMultiplication.algorithm_2(*matrices)
        elif algorithm == 3:
            MatrixMultiplication.algorithm_3(*matrices)
        elif algorithm == 4:
            MatrixMultiplication.algorithm_4(*matrices)
        elif algorithm == 5:
            MatrixMultiplication.algorithm_5(*matrices)
        end_time = time.time()
        total_time = end_time - start_time
        return total_time


    def experiment_3(self):
        data = {"Size": [], "Algorithm-1": [], "Algorithm-2": [], "Algorithm-3": [], "dT1(n)e": [], "dT2(n)e": [], "dT3(n)e": []}
        for size in range(10, 301, 10):
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            C = np.zeros((size, size))
            matrices = (A, B, C, size, size, size)
            times_algo1 = []
            times_algo2 = []
            times_algo3 = []
            for _ in range(10):
                start_time = time.time()
                self.measure_time(1, matrices)
                end_time = time.time()
                times_algo1.append(end_time - start_time)
                
                start_time = time.time()
                self.measure_time(2, matrices)
                end_time = time.time()
                times_algo2.append(end_time - start_time)
                
                start_time = time.time()
                self.measure_time(3, matrices)
                end_time = time.time()
                times_algo3.append(end_time - start_time)
            
            avg_time_algo1 = sum(times_algo1) / len(times_algo1)
            avg_time_algo2 = sum(times_algo2) / len(times_algo2)
            avg_time_algo3 = sum(times_algo3) / len(times_algo3)
            
            data["Size"].append(size)
            data["Algorithm-1"].append(avg_time_algo1)
            data["Algorithm-2"].append(avg_time_algo2)
            data["Algorithm-3"].append(avg_time_algo3)
            
            # Calculate and add the theoretical values
            dT1 = $SELECTION_PLACEHOLDER$
            dT2 = $SELECTION_PLACEHOLDER$
            dT3 = $SELECTION_PLACEHOLDER$
            data["dT1(n)e"].append(dT1)
            data["dT2(n)e"].append(dT2)
            data["dT3(n)e"].append(dT3)
            
        df = pd.DataFrame(data)
        print(df)
        return df
        
    self = MatrixMultiplication('/path/to/input_file.txt')
    self.experiment_3()
    #Method used to execute Experiment II
    def experiment_2(self, algorithm_list):
        data = {"Algorithm": [], "Matrix size": [], "Time": []}
        for algorithm in algorithm_list:
            for i in range(10):
                size = 2 ** i
                A = np.random.rand(size, size)
                B = np.random.rand(size, size)
                C = np.zeros((size, size))
                matrices = (A, B, C, size, size, size)
                time = self.measure_time(algorithm, matrices)
                data["Algorithm"].append(f' Algorithm {algorithm}')
                data["Matrix Size"].append(size)
                data["Time"].append(time)
        df = pd.DataFrame(data)
        print(df)
        return df
        

    #Matrix Multiplication Method
    def matrix_multiplication(A, B):
        C = np.dot(A, B)
        return C
    
    #Method for the matrix chain multiplication experiment
    def matrix_chain_multiplication_experiment(self, input_file):
        matrices = self.read_input_file(input_file)
        results = []

        for chain_id, matrix_chain in enumerate(matrices, start = 2):
            start_time = time.time()
            end_time = time.time()
            total_time = end_time - start_time
            results.append((chain_id, len(matrix_chain) * 10, total_time))

        df = pd.DataFrame(results, columns = ["Chain ID", "Matrix Size", "Time"])

        return df


 # def generate_square_matrices(self, sizes):
 #       matrices = []
 #       for size in sizes:
 #           A = np.random.rand(size, size)
 #           B = np.random.rand(size, size)
 #           matrices.append((A, B))
 #       return matrices

 #   def experiment_1(self, algorithm_list, matrix_sizes):
 #       data = {"Algorithm": [], "Matrix Size": [], "Time": []}
 #       for algorithm in algorithm_list:
 #           for size in matrix_sizes:
 #               matrices = self.generate_square_matrices([size])  # Generate pairs of square matrices
 #               A, B = matrices[0]  # Assuming each pair contains two matrices A and B
 #               start_time = time.time()
 #               result = self.matrix_multiplication(A, B)  # Assuming matrix_multiplication is defined
 #               end_time = time.time()
 #               time_taken = end_time - start_time
 #               data["Algorithm"].append(f"Algorithm {algorithm}")
 #               data["Matrix Size"].append(size)
 #               data["Time"].append(time_taken)
 #       df = pd.DataFrame(data)
 #       return df
def experiment_2(self, algorithm_list):
    data = {"Algorithm": [], "Matrix Size": [], "Time": []}
    for algorithm in algorithm_list:
        for i in range(10):
            size = 2 ** i
            matrices = self.generate_square_matrices([size])  # Generate pairs of square matrices
            A, B = matrices[0]  # Assuming each pair contains two matrices A and B
            start_time = time.time()
            result = self.measure_time(algorithm, (A, B))  # Measure the time for the specified algorithm
            end_time = time.time()
            time_taken = end_time - start_time
            data["Algorithm"].append(f"Algorithm {algorithm}")
            data["Matrix Size"].append(size)
            data["Time"].append(time_taken)
    df = pd.DataFrame(data)
    print(df)
    return df

algorithm_list = [1, 3, 4, 5]
self.experiment_1(algorithm_list)

def generate_matrix_chains(self):
    matrix_chains = []
    for j in range(2, 21):
        dimensions = []
        for i in range(10):
            p = np.random.randint(10, j * 10)
            dimensions.append((p, p + 1))
        matrix_chains.append(dimensions)
    print(matrix_chains)

t.show()


