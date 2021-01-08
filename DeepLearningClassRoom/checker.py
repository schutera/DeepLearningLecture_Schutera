'''
 _______ _______ _______ _______      _______ _     _ _____ _______ _______
    |    |______ |______    |         |______ |     |   |      |    |______
    |    |______ ______|    |         ______| |_____| __|__    |    |______

'''

import numpy as np

# Test for the basic add function.
def test_add(add):
    assert add(40, 2) == 42, "40 + 2 should be 42"
    assert add(9, -2) == 7, "9 - 2, should be 7"
    assert add(5.9, 2.1) == 8, "5.9 + 2.1 should be 8"
    assert add(9, 0) == 9, "9 + 0 should be 9"
    assert add(5, 5) == 10, "5 + 5 should be 10"
    print("Everything passed, you are ready to go.")


def test_forwardPass(forwardPass):
    assert forwardPass(np.array([[0.2], [4]]), np.array([[0.1, 0.5]])) == 4.0804, "Given x and W  the output signal should be 4.0804"
    print("Forward Pass was successful, you are ready to go.")


def test_objectiveFunction(objectiveFunction):
    assert objectiveFunction(0, 1) == 1, "Given prediction 0 and label 1 the loss should be 1"
    assert objectiveFunction(0, 0) == 0, "Given prediction 0 and label 0 the loss should be 0"
    assert objectiveFunction(0.5, 1) == 0.25, "Given prediction 0.5 and label 1 the loss should be 0.25"
    assert objectiveFunction(0.31415926, 1) == 0.47037752064374755, "Given prediction 0.31415926 and label 1 the loss should be 0.47037752064374755"
    print("Your Objective Function was successful, you are ready to go.")

def test_gradientFunction(gradientFunction):
    assert gradientFunction( np.array([[2.0]]), np.array([[0.2], [4]]), 3, 2).all() == np.array([[1.6, 32.]]).all(), "You broke it, check the dimensions."
    print("Gradient Function was successful, you are ready to go.")

def test_update(update):
    assert update(np.array([[ 1.6, 32. ]]), np.array([[ 1., 3. ]]), 1e-5).all() == np.array([[ 1.59999, 31.99997]]).all(), "Not quite right, the convention is to substract from the current weight."
    print("Update function works fine, you are ready to go.")

def test_normalize(x, x_norm):
    assert x_norm.all() == (x/255).all(), "A normalization is done by dividing each value by the max possible value. Which in our case is?"
    print("Normalization worked out well, you are ready to go.")






# Copyright and contact: Mark.schutera@mailbox.org

