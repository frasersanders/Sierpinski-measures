### python version 3.10.5

import numpy as np
import math
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime.now()

### These two maps are used to generate the no.s mod 2^n in an order which
### preserves congruence classes modulo 2^k for all k
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

    return new_array #2-D numpy array

def gen_numbers(n):
    ### Using the two maps given above this returns an array containing the
    ### numbers 1,...,2^n-1 in an order determined by their binary rep'n
    numbers = np.array([0,1])
    while len(numbers) < 2**n :
        numbers = np.concatenate((f_0(numbers),f_1(numbers)))
    return numbers #1-D numpy array

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
    numbers = gen_numbers(n)
    for c in maps:
        for i in numbers:
            if i == 0:
                temp_bool = (numbers == (i+c)%(2**n)).astype("int64")
            else:
                temp_bool = np.vstack((temp_bool,(numbers == (i+c)%(2**n))))
        adj_array += temp_bool # bools can be added, True=1, False=0
    return adj_array #2-D numpy array

def M_upto_n(n,p,q):
    ### Given n, p, q this outputs an array [M_1, M_2^(1/2),...,M_n^(1/n)]
    array = np.array([])

    adj_array = gen_adj_array(n,p,q)[:,0:2**(n-1)]
    ## We only actually need the left half of adj_array so we slice it

    matrix_list = [] ## Creating an list of the matrices we need to multiply
    for m in range(n) :
        matrix_list = [adj_array]+matrix_list
        adj_array  = gen_new_array(adj_array)
        
    for j in range(len(matrix_list)): ## Multiplying the matrices and adding
        ## the first entry of the resulting vector to the array which we
        ## then output. NB @ is shorthand for matrix multiplication.
        if j==0:
            matrix_product = matrix_list[0]
        else:
            matrix_product = matrix_list[j] @ matrix_product
        array = np.append(array, matrix_product[0,0]**(1/(j+1)))

    return array #1-D numpy array with n elements

N=7 ##maximum n which we plot
x = np.arange(1, N+1, 1)

for q in range(2**N): ##iterating through all 0<= q < 2^N
    print(q)
    for p in range(1,int(q/2)+1): ##iterating through 0 < p < q/2
        if math.gcd(p,q)==1: # and not (p%4==0 or q%4==0 or (q-p)%4==0):
            ##checking that p/q is a fraction in its simplest form
            plt.plot(x, M_upto_n(N,p,q))
            ##plotting M_1(p,q),...,M_N(p,q) for n=1,...,N

plt.title(r"$\mathcal{M}_n(p,q)$ for $p,q<2^"+str(N)+"$")
plt.xlabel(r"$n$")
plt.ylabel(r"$\mathcal{M}_n^{(1/n)}$")
plt.xticks(x)

print(start)
string=""
for i in str(start):
    if i not in "- :.":
        string+=i
print(string)
filename=string+".png"
plt.savefig(filename)
##saving the graph as a png using a filename based on the date and time at
##which the program was run

end = datetime.datetime.now()
print(end-start) ##prints the time that the program took to run
plt.show() ##opens matplotlib and displays the graph

            
