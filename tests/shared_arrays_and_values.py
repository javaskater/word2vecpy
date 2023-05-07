#!/usr/bin/python3
import numpy as np
from multiprocessing import Array, Value

from time import sleep
from multiprocessing.pool import Pool

import warnings

def init_shared_variables():
    tmp = np.array([1.0, 2.0, 3.4, 4.5])
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    global_word_count = Value('i', 0) # shared integer ('i') value between processes intialized at 0
    
    return (syn0, global_word_count)

# see https://superfastpython.com/multiprocessing-pool-initializer/
# task executed in a worker process
def task():
    global_word_count.value += 2
    syn0[0] += 1.0
    # report a message
    print('Worker executing task...', flush=True)

# initialize a worker in the process pool
def initialize_worker(*args):
    global syn0, global_word_count
    #syn0_tmp, global_word_count = args[:]
    syn0, global_word_count = args[:]
    #see https://www.pythonpool.com/suppress-warnings-in-python/ for supressing warnings for specific lines of code
    #with warnings.catch_warnings:
     #   warnings.simplefilter('ignore', RuntimeWarning)
      #  syn0 = np.ctypeslib.as_array(syn0_tmp)
    # report a message
    print('Initializing worker...', flush=True)
 
# protect the entry point
if __name__ == '__main__':
    syn0, global_word_count = init_shared_variables()
    # create and configure the process pool
    pool = Pool(processes=2, initializer=initialize_worker, initargs=(syn0, global_word_count))
    # issue tasks to the process pool
    for _ in range(2):
        pool.apply_async(task)
        # close the process pool
    pool.close()
    # wait for all tasks to complete
    pool.join()

    print(f"le tableau est devenu {'|'.join([str(t) for t in syn0])}")
    print(f"le global word count vaut {global_word_count.value}")
