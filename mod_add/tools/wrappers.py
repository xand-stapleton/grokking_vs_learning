import functools


# basic Convolutional Neural Network for input datapoints that are 2D tensors (matrices)
# size of input is (num_datapts x input_channels x input_dim x input_dim)  and requires input_dim % 4 = 0
# Conv, Pool, Conv, Pool, Fully Connected, Fully Connected, ...,  Output
# zero padding is included to ensure same dimensions pre and post convolution
def store_init_args(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Store the arguments in a dictionary
        self._init_args = (args, kwargs)

        return method(self, *args, **kwargs)

    return wrapper
