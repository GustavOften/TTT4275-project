import numpy as np

def format_matrix_for_latex(matrix):
    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            print(str(matrix[i,j]) + " & ", end = '')
        print("\\\\")



