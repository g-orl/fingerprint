import numpy

def load_data(
        filename: str, 
        m: int):
    data = []
    lables = []
    f = open(filename, 'r')
    for line in f:
        values = line.split(',')
        data.append(numpy.array([float(i) for i in values[0:m]], dtype=float))
        lables.append(int(values[m]))
    return (numpy.array(data).T, numpy.array(lables).astype(int))

def vrow(v):
    return v.reshape(1, len(v))

def vcol(v):
    return v.reshape(len(v), 1)