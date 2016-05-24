import numpy as np

def expect_equal(value1, value2):
    # value1, value2 are numpy.ndarray (or like)
    if np.array_equal(value1, value2):
        print "Passed!"
    else:
        print "Failed!"

def expect_inrange(value, _range):
    if value in _range:
        print "Passed!"
    else:
        print "Failed!"
