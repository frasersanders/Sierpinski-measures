import numpy as np
import math
import matplotlib.pyplot as plt

numbers = np.array([0,1])

n=6

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
        #print()
        #print((matrix_product[0].T/9**index))
    #print("M_"+str(index)+"^1/"+str(index)+" :")
    #print(np.amax(matrix_product)**(1/index))
    return np.power(np.amax(matrix_product),(1/index))

data = np.full((2**n,2**n),np.nan)

#print("M_"+str(n)+"^1/"+str(n)+" :")

for q in range(1,2**n):
    print(q)
    for p in range(1,2**n):
        if (p%2==1 or q%2==1):
            new_point = M(p, q)
            data[p,q] = new_point
print()
minimum = np.nanmin(data)

##for q in range(1,2**n):
##    print(q)
##    for p in range(1,int(q/2)+1):
##        if math.gcd(p,q)==1 and not (p%4==0 or q%4==0 or (q-p)%4==0):
##            print("({},{})".format(p,q))
##            data[p,q]=np.nan

plt.imshow(data)
plt.title("M_{}(p,q)^(1/{})".format(n,n))
plt.ylabel("p (mod 2^{})".format(n))
plt.xlabel("q (mod 2^{})".format(n))
plt.xticks(np.arange(0,2**n,2**(n-3)))
plt.colorbar()
print("Maximum M_{}:{}".format(n,np.nanmax(data)))
print("Minimum M_{}:{}".format(n,np.nanmin(data)))
print(np.count_nonzero(data<4.65))
plt.show()
