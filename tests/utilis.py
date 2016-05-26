import numpy as np

def expect_equal(value1, value2):
    # value1, value2 are numpy.ndarray (or like)
    if np.array_equal(value1, value2):
        print "Passed!"
    else:
        print "Failed!"

def expect_not_equal(value1, value2):
    # value1, value2 are numpy.ndarray (or like)
    if np.array_equal(value1, value2):
        print "Failed!"
    else:
        print "Passed!"

def expect_inrange(value, _range):
    # value is a scalar
    if value in _range:
        print "Passed!"
    else:
        print "Failed!"
