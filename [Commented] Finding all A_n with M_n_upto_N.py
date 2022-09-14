### python version 3.10.5

import numpy as np
import math

#These two maps are used to generate the no.s mod 2^n in an order which
#preserves congruence classes modulo 2^k for all k
def f_0(x):
    return 2*x

def f_1(x):
    return 2*x+1

def gen_new_array(array):
    ### Given a matrix with an even number of rows and columns, this returns a
    ### matrix half the size with elements that are half the sum of each 2 by 2
    ### subblock
    new_array = np.array([])
    for i in range(int(len(array)/2)):
        new_line = []
        for j in range(int(len(array[0])/2)):
            somme = array[2*i,2*j]+array[2*i,2*j+1]
            new_line.append(somme)
        if i==0:
            new_array = np.array(new_line)
        else:
            new_array = np.vstack((new_array,new_line))

    return new_array

def gen_adj_array(n, p, q):
    ### Given n, p, q this generates A_n i.e. the adjacency matrix of the
    ### graph with vertices 0,...,2^n-1 with directed edges between vertices
    ### i and j iff there exists an a-b (a, b in {0,p,q}) s.t 
    ### i = j + (a-b) (mod 2^n)
    maps = []
    for i in [0,p,q]:
        for j in [0,p,q]:
            maps.append(i-j)
            
    adj_array = np.zeros((2**n,2**n), dtype="int64")

    for c in maps:
        for i in numbers:
            if i == 0:
                temp_bool = (numbers == (i+c)%(2**n)).astype("int64")
            else:
                temp_bool = np.vstack((temp_bool,(numbers == (i+c)%(2**n))))

        adj_array += temp_bool
    return adj_array

def M_upto_n(n,p,q):
    ### Given n, p, q this outputs an array [M_1, M_2^(1/2),...,M_n^(1/n)]
    array = np.array([])

    adj_array = gen_adj_array(n,p,q)[:,0:2**(n-1)]

    matrix_list = []

    for m in range(n) :
        matrix_list = [adj_array]+matrix_list
        adj_array  = gen_new_array(adj_array)

    for j in range(len(matrix_list)):
        if j==0:
            matrix_product = matrix_list[0]
        else:
            matrix_product = matrix_list[j] @ matrix_product
        array = np.append(array, matrix_product[0,0])

    return(array)

n=4
matrix_list = []
count=0

numbers = np.array([0,1])
while len(numbers) < 2**n :
    numbers = np.concatenate((f_0(numbers),f_1(numbers)))
print(numbers)
##Generating the numbers 0,...,2^n-1 in an order determined by their
##binary expansion

for q in range(1,2**n):
    for p in range(int(q/2)+1):
        if math.gcd(p,q)==1: ##Iterating through each pair of p and q,
            ## checking that p/q is a fraction in its simplest form and then
            if True: #not(p%4==0 or q%4==0 or (q-p)%4==0):
                ##This if statement can be changed to consider certain
                ##subsets of the (p,q)
                count+=1 ##counting how many cases we have checked
                new_array = gen_adj_array(n,p,q)[0]
                in_list = False
                for i in matrix_list:
                    if np.array_equal(i, new_array):
                        in_list = True
                        ## Checking if new_array has already been generated
                        ## by a different value of (p,q)
                if not in_list:
                    ## If it hasn't we add it to matrix_list and print the
                    ## following data about (p,q)
                    matrix_list.append(new_array)
                    print("({},{}):".format(p,q))
                    print("First row of A_{}:".format(n))
                    print(new_array)
                    print("M_n from n=1 to n={}".format(n))
                    print(M_upto_n(n,p,q))
                    print()
print("Unique A_{}'s: {}".format(n, np.array(matrix_list).shape[0]))
a = np.array([1,2])
        
