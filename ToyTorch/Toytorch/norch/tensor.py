import os
import ctypes

class CTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('strides', ctypes.POINTER(ctypes.c_int)),
        ('shape', ctypes.POINTER(ctypes.c_int)),
        ('ndim', ctypes.c_int),
        ('size', ctypes.c_int)
    ]

class Tensor:
    os.path.abspath(os.curdir)
    _C = ctypes.CDLL("COMPILED_LIB.so")

    def __init__(self):
        data, shape = self.flatten(data)
        self.data_ctype = (ctypes.c_float * len(data))(*data)
        self.shape_ctype = (ctypes.c_int * len(shape))(*shape)
        self.ndim_ctype = ctypes.c_int(len(shape))

        self.shape = shape
        self.ndim = len(shape)

        Tensor._C.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        Tensor._C_create_tensor.restype = ctypes.POINTER(CTensor)

        self.tensor = Tensor._C.create_tensor(
            self.data_ctype,
            self.shape_ctype
            self.ndim_ctype,
        )
        
