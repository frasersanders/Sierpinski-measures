import numpy as np
import math
import matplotlib.pyplot as plt
import datetime

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
                data[p,q] = new_point
                data[q,p] = new_point
    print(data)              
    print("Maximum M_{}:{}".format(n,np.nanmax(data)))
    print("Minimum M_{}:{}".format(n,np.nanmin(data)))
    return(data)

N=5
count=0
fig, axs  = plt.subplots(1, 2, sharey = True, sharex=True)
for q in range(2**N):
    for p in range(1,int(q/2)+1):
        if math.gcd(p,q)==1:
            count+=1
            x = np.arange(1, N+1, 1)
            array = np.array([])
            for i in x:
                array = np.append(array, M(i, p, q))
            if not(p%4==0 or q%4==0 or (q-p)%4==0):
                if (p%4==3 or q%4==3 or (q-p)%4==3): #(int(p/2)%4==0 or int(q/2)%4==0 or int((q-p)/2)%4==0):
                    axs[0].plot(x,(array), label  = "({},{})".format(p,q))
                    print("({},{})".format(p,q))
                    print("({},{})".format(q-p,q))
                else: 
                    axs[1].plot(x,(array), label  = "({},{})".format(p,q))

#axs[0].xticks((x))
#axs[1].xticks((x))
print(count)
#plt.legend()
#plt.plot(x, np.power((5**2)*4.5**(x-2),1/(x)), linestyle = "dashed", label = "(5^2×4.5^(n-2))^(1/n)")
#plt.plot(x, np.power(5*4.5**(x-1),1/x), linestyle = "dashed", label = "(5×4.5^(n-1))^(1/n)")
plt.ylim(4.55, 5.05)
start=datetime.datetime.now()
print(start)
string=""
for i in str(start):
    if i not in "- :.":
        string+=i
print(string)
filename="C:\\Users\\frase\\Documents\\Fractal Project\\graphs\\"+string+".png"
plt.savefig(filename)
plt.show()

