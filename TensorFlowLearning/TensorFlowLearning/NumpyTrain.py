#num1
import numpy as np 

a=np.array([4,5,6])

#num2
print(a.dtype)
print(a.shape)
print(a[0])

#num3
b=np.array([[4,5,6],[1,2,3]]).reshape(2,3)
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