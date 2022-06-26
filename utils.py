import numpy as np 

def convert_log_to_simple_return(log_return: float): 

    simple_return = np.exp(log_return) - 1
    return simple_return

def convert_simple_to_log_return(simple_return: float): 

    assert simple_return > -1, "Simple return can't exceed 100%. You entered a value of {:.2f}".format(simple_return)

    log_return = np.log(1 + simple_return) 
    return log_return

