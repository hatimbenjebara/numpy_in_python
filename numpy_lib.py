import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
arr = np.array([1,2,3,4,5])
print(arr)
print(type(arr))
#no different between tuple and list 
arr = np.array((1, 2, 3, 4, 5))
print(arr)
print(type(arr))
#Scalar or 0-D arrays 
scalar = np.array(42)
print(scalar)
print(type(scalar))
#Create a 1D array from a list 
arr1=np.array([1,2,3,4,5])
print(arr1)
#Create a 2D array from a list of lists
arr2= np.array([[1,2,3],[4,5,6]])
print(arr2)
#create 1-D array of zeros with length 5
zeros_array= np.zeros(5)
print(zeros_array)
#create a 2-Darray of ones with shape( 3,4)
ones_arr = np.ones((3,4))
print(ones_arr)
#create a 1D array of integers from 0 to 4
int_arr = np.arange(5)
print(int_arr)
#get the first element of arr
x=arr1[0]
print(x)
#get the first row of arr
row = arr2[0,:]
print(row)
#set the second element of arr to 10 
arr2[1] = 10
print(arr2)
#create a matrix using numpy.matrix()
mat1 = np.matrix([[1,2], [3,4]])
print(mat1)
#create a matrix using numpy.array() with dtype = ‘object’
mat2 = np.array([[1,2],[3,4]], dtype = 'object')
print(mat2)
#create a matrix using numpy.array() and cast it as a matrix
mat3 = np.array([[1,2],[3,4]]).astype(np.matrix)
print(mat3)
#check dimension of array 
arr = np.array([1,2,3, 4, 5])
print(arr.ndim)
#to shape using ndim
arr = np.array([1,2,3, 4, ], ndmin=5)
print(arr)
#reshape
a= np.array([1, 2, 3, 4, 5, 6])
b=a.reshape((2,3))
print(b)
new_arr= a.reshape(2,3,-1)
print(new_arr)
#flatten array
arr= np.array([[1,2,3], [4,5,6]])
print(arr)
#flatten the array using flatten() method
flat_arr = arr.flatten()
print("Original array:") 
print( arr)
print("flattened array: ")
print(arr)
# flatten array using ravel() method 
import numpy as np
arr= np.array([[1,2,3], [4,5,6]])
#flatten the array
flat_arr = arr.ravel()
print("Original array: ") 
print( arr)
print("flattened array: ")
print(arr)
#create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
#iterate over the elements using a for loop
for i in range(arr.shape[0]):
	for j in range(arr.shape[1]):
		print(arr[i][j])
#Iterate over the element using nditer()
for element in np.nditer(arr):
	print(element) 
#iterate over the elements and their indices using ndenumerate()
for index, element in np.ndenumerate(arr):
	print(index, element)
#join two array using concatenate() method
#create two array
arr1 = np.array([[1,2], [3,4]]) 
arr2=np.array([[5,6]])
#concatenate the arrays along axis 0 
result = np.concatenate((arr1,arr2), axis=0)
print( result)
#join two array using stack() method 
#create two array
arr1 = np.array([1,2,3]) 
arr2=np.array([4,5,6])
#stack the arrays along new axis  
result = np.stack((arr1,arr2), axis=0)
print( result)
#search inside array using where
#create an array 
arr = np.array([1, 2, 3, 4, 5, 6])
#find the indices of elements
indices = np.where( arr==2)
print(indices) 
#search inside array using searchsorted
#create a sorted array
arr = np.array([1, 2, 3, 4, 5, 6])
#find the index where the value 3 should be inserted 
index = np.searchsorted(arr,3)
print(index)
#sort array using sort() method
#create an array 
arr = np.array([3, 1, 4, 2, 5])
#sort the array in ascending order
arr.sort()
print(arr)
#sort array using sort() method and choose the axis
#create an array 
arr = np.array([[3, 2, 1], [4, 6, 5]])
#sort the array in ascending order
arr.sort(axis=0)
print(arr)
#sort array using argsort()
#create an array 
arr = np.array([3, 1, 4, 2, 5])
#get the indices that would sort the array
indices = np.argsort(arr)
print(indices) 
#sort the array using the indices 
sorted_arr = arr[indices]
print(sorted_arr)
#filtring array
#example1, 
arr= np.array([41,42,43,44])
#create empty list
filter_arr=[]
#go through each element in arr
for element in arr:
#if the element is higher than 42, set the value to True, otherwise False: 
	if element > 42: 
		filter_arr.append(True)
	else:
		filter_arr.append(False) 
new_arr= arr[ filter_arr] 
print(new_arr)
#example 2: filter directly
arr = np.array([41, 42, 43, 44])
filter_arr = arr>42
print(arr)
#example3, 
arr = np.array([1,2,3,4,5,6,7])
#create an empty list 
filter_arr = []
#go through each element in arr 
for element in arr: 
#if the element is completely divisible by 2, set the value to True, otherwise False
	if element%2 == 0:
		filter_arr.append(True)
	else:
		filter_arr.append(False)
new_arr = arr[filter_arr]
print(new_arr)
#example4 : 
arr = np.array([1, 2, 3, 4, 5, 6, 7])
filter_arr = arr %2 ==0
new_arr = arr[filter_arr]
print(new_arr)
#random using numpy.random.rand
random_array = np.random.rand(3,3)
print(random_array) 
#random using numpy.random.randn
random_array = np.random.randn(3,3)
print(random_array)
#random using numpy.random.randint
random_array = np.random.randint(1, 10, size=(3,3))
print( random_array)
#example2 ,
from numpy import random 
x= random.randint(100)
print(x) 
#random using choice() method
#example1: generate a random sample of size 3 from an array
a = np.array([1, 2, 3, 4, 5])
random_sample = np.random.choice(a, size=3)
print(random_sample) #output: [2 5 4]
#Example 2: Generate a random sample of size 3 without replacement 
a = np.array([1, 2, 3, 4, 5])
random_sample = np.random.choice(a, size = 3, replace= False)
print( random_sample) #output: [3 1 4]
#Example 3: Generate a random sample of size 3 with custom probabilities
a= np.array([1, 2, 3, 4, 5])
p = [0.1, 0.2, 0.3, 0.2, 0.2] #probabilities associated with each element in a 
random_sample = np.random.choice(a, size=3, replace= False, p=p)
print(random_sample) #output : [ 2 1 3]
#data distribution
#uniform distribution
random_numbers = np.random.uniform(low=0, high=1, size=(3,3))
print(random_numbers)
random_numbers = np.random.uniform(low=0, high=1, size=10000)
sns.distplot(random_numbers, hist = False)
plt.title('uniform Distribution of random numbers')
plt.show()
#normal distribution 
random_numbers = np.random.normal(loc=0, scale=1, size=(3,3))
print(random_numbers)
random_numbers = np.random.normal(loc=0, scale=1, size=10000)
sns.distplot(random_numbers, hist = False)
plt.title('normal Distribution of random numbers')
plt.show()
#binomial distribution 
random_numbers = np.random.binomial(n=10, p=0.5, size=(3,3))
print(random_numbers)
random_numbers = np.random.binomial(n=10, p=0.5, size=100000)
sns.distplot(random_numbers, hist = False)
plt.title('binomial Distribution of random numbers')
plt.show()
#difference between normal and binomial distribution :
sns.kdeplot(np.random.normal(loc=0, scale=1, size=100000), label="normal distribution")
sns.kdeplot(np.random.binomial(n=4, p=0.5, size = 100000), label="binomial distribution")
plt.title("Difference between normal and binomial distribution")
plt.legend()# print names 
plt.show()
#poisson distribution 
random_numbers = np.random.poisson(lam=5, size=(3,3))
print(random_numbers)
random_numbers = np.random.poisson(lam=5, size=100000)
sns.distplot(random_numbers, hist = False)
plt.title('poisson Distribution of random numbers')
plt.show()
#difference between normal and poisson distribution 
sns.distplot(random.normal(loc=0, scale=1, size=100000), hist=False, label="normal distribution")
sns.distplot(random.poisson(lam=5, size = 100000), hist= False, label= "poisson distribution")
plt.title("difference between normal and poisson distribution")
plt.legend()
plt.show()
#difference between poisson and binomial distribution
sns.distplot(np.random.binomial(n=4, p=0.5, size= 100000), hist=False, label= "binomial distribution")
sns.distplot(np.random.poisson(lam=5, size= 100000), hist=False, label="poisson distribution")
plt.title("difference between poisson and binomial distribution")
plt.legend()
plt.show()
#logistic distribution
#set the parameter of the logistic distribution
m = 5.0 # location parameter
s = 1.0# scale parameter
#generate a logistic distribution with 1000 samples
samples = np.random.logistic(m,s, size= 1000)
print("this is logistic distribution :", samples)
#plot a histogram of the samples
plt.hist(samples, bins=50, density= True)
sns.distplot(samples, hist=False, label='Logistic distribution')
plt.title("Logistic distribution")
plt.xlabel("x-axis")
plt.ylabel("Density")
plt.legend()
plt.show()
plt.show()
#different between normal and logistic distribution
sns.distplot(random.normal(scale=2, size=1000), hist = False, label = "Normal distribution")
sns.distplot(random.logistic(size =1000), hist=False, label="Logistic distribution")
plt.title("Difference between Normal and Logistic distribution")
plt.xlabel("x-axis")
plt.ylabel("Density")
plt.legend()
plt.show()
#multinomial distribution 
n_trials =10
pvals = [0.2, 0.3, 0.5]
n_samples = 5
samples= np.random.multinomial(n=n_trials, pvals=pvals, size= n_samples)
print("multinomial distribution : ", samples)
fig, ax = plt.subplots()
ax.bar(range(len(pvals)), samples[0], label="Sample 1")
ax.bar(range(len(pvals)), samples[1], bottom=samples[0], label="Sample2")
ax.bar(range(len(pvals)), samples[2], bottom=samples[0]+samples[1], label="Sample3")
ax.bar(range(len(pvals)), samples[3], bottom=samples[0]+samples[1]+samples[2], label="Sample4")
ax.bar(range(len(pvals)), samples[4], bottom=samples[0]+samples[1]+samples[2]+samples[3], label="Sample5")
ax.set_xticks(range(len(pvals)))
ax.set_xticklabels(["Outcome1", "Outcome2", "Outcome3"])
ax.set_xlabel("Outcomes")
ax.set_ylabel("Counts")
ax.set_title(f"Multinomial Distribution (n={n_trials}, p={pvals}, samples={n_samples})")
ax.legend()
plt.show()
#Ray leigh distribution 
sigma = 2
n_samples=1000
samples = np.random.rayleigh(scale=sigma, size=n_samples)
fig , ax = plt.subplots()
ax.hist(samples, bins=30, density= True)
ax.set_xlabel("x")
ax.set_ylabel("Probability density function ")
ax.set_title(f"Rayleigh distribution ( sigma={sigma}, samples={n_samples})")
plt.show() 
sns.distplot(np.random.rayleigh(size=1000), hist= False)
plt.title("Rayleigh distribution")
plt.xlabel("X-axis")
plt.ylabel("Density")
plt.show()
#perato distribution 
samples = np.random.pareto(a=3, size=1000)
print("Mean: ", np.mean(samples))
print("Median:", np.median(samples))
print("Standard deviation:", np.std(samples))
plt.hist(samples, bins=50, density=True, alpha=0.5)
x=np.linspace(0.01, 10, 1000)
y = 3 * (x**(-4))
plt.plot(x,y,'r-', lw=2)
plt.xlabel("x")
plt.ylabel("Probability density")
plt.title("Pareto distribution with shape parameter a=3")
plt.show() 
#zipf distribution 
samples = np.random.zipf(a=2, size=1000)
print("Mean:", np.mean(samples))
print("Median: ", np.median(samples))
print("Standard deviation: ", np.std(samples))
samples = np.random.zipf(a=2, size=1000)
values, counts = np.unique(samples, return_counts=True)
freqs = counts/len(samples)
fig, ax = plt.subplots()
ax.loglog(values, freqs, 'bo')
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=10)
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Zipf distribution with shape parameter a=2")
plt.show()



#exponential distribution 
random_numbers = np.random.exponential(scale=2, size=(3,3))
print(random_numbers)
random_numbers = np.random.exponential(scale=2, size=100000)
sns.distplot(random_numbers, hist = False)
plt.title('exponential Distribution of random numbers')
plt.show()
#gamma distribution 
random_numbers = np.random.gamma(shape=2, scale=2, size=(3,3))
print(random_numbers)
random_numbers = np.random.gamma(shape=2, scale=2, size=100000)
sns.distplot(random_numbers, hist = False)
plt.title('gamma Distribution of random numbers')
plt.show()
#beta distribution 
random_numbers = np.random.beta(a=2, b=5, size=(3,3))
print(random_numbers)
random_numbers = np.random.beta(a=2, b=5, size=100000)
sns.distplot(random_numbers, hist = False)
plt.title('beta Distribution of random numbers')
plt.show()
#chi square distribution 
random_numbers= np.random.chisquare( df=3, size=(3,3))
print(random_numbers)
random_numbers= np.random.chisquare( df=3, size=100000)
sns.distplot(random_numbers, hist = False)
plt.title('chi square Distribution of random numbers')
plt.show()
#geometric distribution 
random_numbers= np.random.geometric( p=0.5, size=(3,3))
print(random_numbers)
random_numbers= np.random.geometric( p=0.5, size=100000)
sns.distplot(random_numbers, hist = False)
plt.title('geometric Distribution of random numbers')
plt.show()
#weibull distribution 
random_numbers= np.random.weibull( a=2, size=(3,3))
print(random_numbers)
random_numbers= np.random.weibull( a=2, size=100000)
sns.distplot(random_numbers, hist = False)
plt.title('weibull Distribution of random numbers')
plt.show()

#permutation 
arr = np.arange(5) 
permutations = np.random.permutation(arr)
print("original array:", arr)
print("permutation: ", permutations)
#shuffle 
arr = np.arange(5) 
shuf = np.random.shuffle(arr)
print("shuffled array:", arr)

#universal function 
x = np.linspace(0,2*np.pi, 100)
y = np.sin(x)
print(y)
# trigno
arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
y= np.sin(arr)
plt.plot(arr, y)
plt.xlabel("angle(radian)")
plt.ylabel("sine")
plt.title("sine wave")
plt.show()










