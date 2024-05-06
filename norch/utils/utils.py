import random

def generate_random_list(shape):
    """
    Generate a list with random numbers and shape 'shape'
    """
    if len(shape) == 0:
        return []
    else:
        inner_shape = shape[1:]
        if len(inner_shape) == 0:
            return [random.random()] * shape[0]
        else:
            return [generate_random_list(inner_shape) for _ in range(shape[0])]
        