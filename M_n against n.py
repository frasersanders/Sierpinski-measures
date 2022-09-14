import numpy as np
import math
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime.now()

def f_0(x):
    return 2*x

def f_1(x):
    return 2*x+1

def gen_new_array(array):
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
    numbers = np.array([0,1])
    while len(numbers) < 2**n :
        numbers = np.concatenate((f_0(numbers),f_1(numbers)))
    return numbers

def gen_adj_array(n, p, q):
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

##def M(n,p,q):
##
##    adj_array = gen_adj_array(n,p,q)[:,0:2**(n-1)]
##
##    matrix_list = []
##
##    for m in range(n) :
##        matrix_list = [adj_array]+matrix_list
##        adj_array  = gen_new_array(adj_array)
##
##    print(matrix_list)
##
##    for j in range(len(matrix_list)):
##        if j==0:
##            matrix_product = matrix_list[0]
##        else:
##            matrix_product = matrix_list[j] @ matrix_product
##
##        print(matrix_product)
##
##    return(matrix_product[0,0])

def M_upto_n(n,p,q):
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
        array = np.append(array, matrix_product[0,0]**(1/(j+1)))

    return(array)

N=7
x = np.arange(1, N+1, 1)

for q in range(2**N):
    print(q)
    for p in range(1,int(q/2)+1):
        if math.gcd(p,q)==1: # and not (p%4==0 or q%4==0 or (q-p)%4==0):
            plt.plot(x, M_upto_n(N,p,q))

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

end = datetime.datetime.now()
print(end-start)
plt.show()

            
