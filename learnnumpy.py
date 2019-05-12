import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)

# print(a[...,1:])
#
# print(a[0,1])
#
# print(a[...,1])
#
# print(a[2,...])

# print('\n')
#
# print(a[[[0,1],[0,2]],[[1,2],[2,0]]])
#
# print('\n')
#
# print(a[0:,1:2])

# print(a>5)
#
# print(a[a>5])
#
# print(a[~(a==4)])

b=np.arange(56).reshape(7,8)

# print(b)

# print('\n')

# print(b[0])
# print('\n')
# print(b[[3,1,0]])
# print(b[[2]])
# print('\n')
# print(b[...,[2,0]])
#
# print(b[np.ix_([0],[1])])
# print(b[[0],[1]])

# for i in np.nditer(b,op_flags=['readwrite']):
#     i[...]=i*3
#     # print(i*2)
# print(b)

# print(b.reshape(8,7))

# x=np.arange(12).reshape(12,1)
# print(x)
# print('\n')
# y=np.arange(0,120,10).reshape(1,12)
# print(y)
# print('\n')

# z=np.broadcast(x,y)
# print(z.shape)
# i,j=z.iters
# while(i.next()&j.next()):
#     print(i.next(),j.next())
# print(x+y)

# print(np.broadcast_to(x,(12,12)))

# c=np.expand_dims(a,axis=0)
# print(c)
# print(c.shape)
# print(c.ndim)
#
# print(np.squeeze(c))
# print(np.squeeze(c).shape)

d=np.arange(12).reshape(3,2,2)
print(d)
print('\n')
e=np.delete(d,0,axis=2)
print(e)
print(e.shape)