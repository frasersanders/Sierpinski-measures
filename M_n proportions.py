import numpy as np
import math
import matplotlib.pyplot as plt

def f_0(x):
    return 2*x

def f_1(x):
    return 2*x+1

def gen_new_array(array):
    new_array = np.array([])
    for i in range(int(len(array)/2)):
        temp_array = np.array([])
        for j in range(int(len(array)/2)):
            somme = array[2*i,2*j]+array[2*i,2*j+1]#+array[2*i+1,2*j]+array[2*i+1,2*j+1]
            temp_array = np.hstack((temp_array,[somme])) #/2]))

        if i == 0:
            new_array = temp_array
        else:
            new_array = np.vstack((new_array,temp_array))

    return new_array

def copy_matrix(matrix, k): 
    new_matrix = matrix
    for i in range(k):
        new_matrix = np.block([[new_matrix,np.zeros(new_matrix.shape)],
                               [np.zeros(new_matrix.shape),new_matrix]])#.astype("int32")
    return new_matrix #a block matrix with 2^k copies along the diagonal

def gen_numbers(n):
    numbers = np.array([0,1])
    while len(numbers) < 2**n :
        numbers = np.concatenate((f_0(numbers),f_1(numbers)))
    return numbers

def M(n,p,q):
    maps = []
    for i in [0,p,q]:
        for j in [0,p,q]:
            maps.append(i-j)

    adj_array = np.zeros((2**n,2**n), dtype="int64")
    numbers  = gen_numbers(n)
    
    for c in maps:
        for i in numbers:
            if i == 0:
                temp_bool = (numbers == (i+c)%(2**n)).astype("int64")
            else:
                temp_bool = np.vstack((temp_bool,(numbers == (i+c)%(2**n))))

        adj_array += temp_bool
    #print()
    #print(adj_array)

    matrix_list = []

    for m in range(n) :
        #print(adj_array)
        matrix_list = [copy_matrix(adj_array,m)] + matrix_list
        adj_array  = gen_new_array(adj_array)
        #print()

    #print(matrix_list)

    matrix_product  = np.eye(2**n).astype("int64")

    index = 0

    for matrix in matrix_list:
        index+=1
        matrix_product = np.matmul(matrix, matrix_product)
        
    return np.amax(matrix_product)**(1/index)

def gen_data(n):
    data = np.full((2**n,2**n),np.nan)
    for q in range(2**n):
        print(q)
        for p in range(q):
            if math.gcd(p,q)==1:
                new_point = M(n, p, q)
                if new_point <= 9:    #Have removed the gcd(p,q)=1 condition since this is unnecessary (consider the first 2^n +1 primes mod 2^n and use the pigeonhole principle)
                    data[p,q] = new_point
                    data[q,p] = new_point
    print(data)              
    print("Maximum M_{}:{}".format(n,np.nanmax(data)))
    print("Minimum M_{}:{}".format(n,np.nanmin(data)))
    return(data)

def percentage_graph(data, n):
    x = np.arange(4.5, 5, 0.001)
    array = np.array([])
    for i in x:
        array = np.append(array, np.count_nonzero(data<i))
    array = 100*array/np.count_nonzero(~np.isnan(data))
    print(np.count_nonzero(~np.isnan(data)))
    
    plt.plot(x, array, label = "n={}".format(n))
    plt.title("Percentage of (p,q) with small M_n")
    plt.ylabel("% (p,q)<2^n s.t. M(p,q)<4.5+??")
    plt.xlabel("4.5+??")
    plt.legend()

def log_log_graph(data, n):
    x = np.arange(4.5, 5, 0.001)
    array = np.array([])
    for i in x:
        array = np.append(array, np.count_nonzero(data<i))
    array = array
    plt.plot(np.log(x-4.5), np.log(array), label = "n={}".format(n))
    plt.title("ln[#(p,q) with small M_n]")
    plt.ylabel("log #{(p,q)<2^n : M(p,q)<4.5+??}")
    plt.xlabel("log(??)")
    plt.legend()

for n in range(6):
    percentage_graph(gen_data(n+1), n+1)

plt.show()
