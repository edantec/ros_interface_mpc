import numpy as np
from std_msgs.msg import Float64MultiArray, Int8MultiArray, MultiArrayDimension


def numpy_to_multiarray_float64(np_array):
    multiarray = Float64MultiArray()
    multiarray.layout.dim = [
        MultiArrayDimension(label="dim%d" % i, size=np_array.shape[i], stride=np_array.shape[i] * np_array.dtype.itemsize)
        for i in range(np_array.ndim)
    ]
    multiarray.data = np_array.ravel().tolist()
    return multiarray

def numpy_to_multiarray_int8(np_array):
    multiarray = Int8MultiArray()
    multiarray.layout.dim = [
        MultiArrayDimension(label="dim%d" % i, size=np_array.shape[i], stride=np_array.shape[i] * np_array.dtype.itemsize)
        for i in range(np_array.ndim)
    ]
    multiarray.data = np_array.ravel().tolist()
    return multiarray


def multiarray_to_numpy(ros_array):
    dims = [d.size for d in ros_array.layout.dim]
    if dims == []:
        return np.array([])
    out = np.empty(dims)
    out.ravel()[:] = ros_array.data
    return out
    
def listof_numpy_to_multiarray_float64(list):
    return numpy_to_multiarray_float64(np.array(list))

def listof_numpy_to_multiarray_int8(list):
    return numpy_to_multiarray_int8(np.array(list))

def multiarray_to_listof_numpy(ros_array):
    return list(multiarray_to_numpy(ros_array))
