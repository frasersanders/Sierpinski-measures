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

    return new_array

def gen_numbers(n):
    ### Using the two maps given above this returns an array containing the
    ### numbers 1,...,2^n-1 in an order determined by their binary rep'n
    numbers = np.array([0,1])
    while len(numbers) < 2**n :
        numbers = np.concatenate((f_0(numbers),f_1(numbers)))
    return numbers

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

N=5 ##maximum n which we consider
x = np.arange(1, N+1, 1)
best_M_n = np.inf ##initailising a variable which keeps track of the
## smallest M_N the program has come across so far
best_p_q_array = [] ##initialising an array which keeps track of the (p,q)
## with the smallest M_n
for q in range(2**N):
    print(q)
    for p in range(1,int(q/2)+1):
        if math.gcd(p,q)==1: # and (p%4==0 or q%4==0 or (q-p)%4==0):
            y = M_upto_n(N,p,q)
            plt.plot(x, y)
            if  y[-1] == best_M_n: ## If best_M_n is achieved at (p,q)
                ## then we add [p,q] to this array
                best_p_q_array.append([p,q])
            elif y[-1] < best_M_n:
                best_M_n = y[-1] ## If we find a better M_n we update
                ## best_M_n and empty best_p_q_array
                best_p_q_array = [[p,q]]
print(best_M_n)
print(best_p_q_array)
print()

##plt.plot(x,M_upto_n(N,1,2), linestyle = "dashed", label = "(1,2)")
##plt.plot(x,M_upto_n(N,pi%(2**N),e%(2**N)))

print(start)
string=""
for i in str(start):
    if i not in "- :.":
        string+=i
print(string)
filename = string+".png"
plt.savefig(filename)
##saving the graph as a png using a filename based on the date and time at
##which the program was run

end = datetime.datetime.now()
print(end-start)
#plt.legend()
plt.show()

            
