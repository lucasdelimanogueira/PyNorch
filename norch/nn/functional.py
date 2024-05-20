import math

def sigmoid(x):
    return 1.0 / (1.0 + (math.e) ** (-x)) 

def softmax(x, dim=None):
    e = math.e ** x
    s = e.sum(axis=dim, keepdim=True)
    #l = s.log()
    print('x', x)
    print('\n\n')
    #print('log', l)

    print('\n\n')
    #print('x-log', x - l)
    

    #return math.e ** (x - sum.log())