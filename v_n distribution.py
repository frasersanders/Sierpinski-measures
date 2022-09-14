import numpy as np
import math
import matplotlib.pyplot as plt

numbers = np.array([0,1])

n=7

#maps = []

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


while len(numbers) < 2**n :
    numbers = np.concatenate((f_0(numbers),f_1(numbers)))

numbers = np.array(range(2**n))

print(numbers)

##for i in [0,p,q]:
##    for j in [0,p,q]:
##        maps.append(i-j)
##
##print(np.array(maps).reshape((3,3)))

def M(p,q):
    maps = []
    for i in [0,p,q]:
        for j in [0,p,q]:
            maps.append(i-j)
    #print(np.array(maps).reshape((3,3)))

    adj_array = np.zeros((2**n,2**n), dtype="int64")

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
        plt.plot(np.array(range(2**index))*2**(n-index),matrix_product[0,:2**index]**(1/index), label = str(index)) #funky order
        #plt.plot(np.array(range(2**index))*2**(n-index),matrix_product[0][numbers[0::2**(n-index)]]**(1/index), label = str(index)) #"normal" order
        #plt.plot(np.array(range(2**index))*2**(n-index),np.log(matrix_product[0][numbers[0::2**(n-index)]]**1), label = str(index)) #log graph with "normal" order
        #plt.plot(np.array(range(2**index))*2**(n-index),np.log(matrix_product[0,:2**index]**1), label = str(index)) #log graph with funky order
    print("M_"+str(index)+"^1/"+str(index)+" :")
    print(np.amax(matrix_product)**(1/index))
    return np.amax(matrix_product)**(1/index)

p=5
q=19
M(p,q)
plt.axhline(4.5, color = "darkgray", lw = 1)
plt.title("p={}, q={}".format(p,q))
plt.legend()
plt.show()
