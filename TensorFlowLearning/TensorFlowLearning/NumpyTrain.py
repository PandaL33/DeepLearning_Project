#numpy Tranin 
#https://github.com/nndl/exercise/blob/master/warmup/numpy_%20tutorial.ipynb
#num1
import numpy as np
import matplotlib.pyplot as plot

a=np.array([4,5,6])

#num2
print(a.dtype)
print(a.shape)
print(a[0])

#num3
b=np.array([[4,5,6],[1,2,3]])
print(b.shape)
print(b[0,0],b[0,1],b[1,1])
print(b[:,1])

#num4
aa=np.zeros([3,3],int)
print(aa)
bb=np.ones([4,5])
print(bb)
cc=np.eye(4)
print(cc)
dd=np.random.rand(3,2)
print(dd)

#num5
e=np.arange(1,13,1).reshape(3,4)
print(e)
print(e[2,3],e[0,0])

#num6
f=e[0:2,2:4]
print(f)
print(f[:,1])
print(f[0,0])

#num7
g=e[1:3,:]
print(g)
print(g[0,-1])

#num8
h=np.arange(1,7,1).reshape(3,2)
print(h)
print(h.shape)
print(h[[0, 1, 2], [0, 1, 0]]) 

#num9
i=np.arange(1,13,1).reshape(4,3)
ii = np.array([0, 2, 0, 1]) 
print(i[np.arange(4), ii])

#num10
#i[np.arange(4), ii]+=10

#num11 / 12
x=np.array([1, 2])
print(x.dtype)
x=np.array([1.0, 2.0])
print(x.dtype)

#num13
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(x+y)
print(np.add(x,y))

#num14
print(x-y)
print(np.subtract(x,y))

#num15
print(x*y)
print(np.multiply(x,y))
print(np.dot(x,y))

#num16
print(np.divide(x,y))

#num17
print(np.sqrt(x))

#num18 
print(x.dot(y))
print(np.dot(x,y))

#num19
print(np.sum(x))
print(np.sum(x,axis =0))
print(np.sum(x,axis = 1))

#num20
print(np.sum(x))
print(np.sum(x,axis =0))
print(np.sum(x,axis = 1))

#num21
print(x.T)

#num22
print(np.exp(x))

#num23
print(np.argmax(x))
print(np.argmax(x,axis=0))
print(np.argmax(x,axis=1))

#num24
x = np.arange(0, 100, 0.1) 
y=x*x
plot.plot(x,y,'r')

#num25
m = np.arange(0, 3 * np.pi, 0.1)
n=np.sin(m)
plot.plot(m,n,'r')

#num26
print(0*np.nan) 
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3==3**0.1)