#cdef extern from "fvht.h":
#    char* say_hello_c()
#
## create the wrapper code
#def say_hello():
#    return say_hello_c()

cdef extern from "math.h":
    double cos(double arg)

def cos_func(arg):
    return cos(arg)
