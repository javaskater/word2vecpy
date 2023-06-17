#!/usr/bin/python3
import numpy as np
from multiprocessing import Array, Value

from time import sleep
from multiprocessing.pool import Pool
import os

import warnings

def init_shared_variables():
    global syn0_tmp
    tmp = np.array([1.0, 2.0, 3.4, 4.5])
    syn0_tmp = np.ctypeslib.as_ctypes(tmp) # put on C Types to be faster
    syn0_tmp = Array(syn0_tmp._type_, syn0_tmp, lock=False)

    with warnings.catch_warnings(): #what to do with warnings of the block instructions underneath
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp) #shared the memory with the c_types array see https://numpy.org/doc/stable/reference/routines.ctypeslib.html

    global_word_count = Value('i', 0) # shared integer ('i') value between processes intialized at 0
    
    return (syn0, global_word_count)

# see https://superfastpython.com/multiprocessing-pool-initializer/
# task executed in a worker process
def task(pid):
    global_word_count.value += 2
    syn0_g[0] += 1.0
    #see https://docs.python.org/3/library/multiprocessing.html to simulate what is called the pid in pool.map
    num_process = os.getpid()
    num_pprocess = os.getppid()
    # report a message
    print(f"[task_{pid}] Worker executing task... pid {num_process} parent pid {num_pprocess} no execution {num_process - num_pprocess}", flush=True)

# initialize a worker in the process pool
def initialize_worker(*args):
    global syn0_g, global_word_count
    #syn0_tmp, global_word_count = args[:]
    syn0_g, global_word_count = args[:]
    #see https://www.pythonpool.com/suppress-warnings-in-python/ for supressing warnings for specific lines of code
    #with warnings.catch_warnings:
     #   warnings.simplefilter('ignore', RuntimeWarning)
      #  syn0 = np.ctypeslib.as_array(syn0_tmp)
    num_process = os.getpid()
    num_pprocess = os.getppid()

    # report a message
    print(f"[initialize_worker] Initializing worker pid {num_process} parent pid {num_pprocess} no execution {num_process - num_pprocess}...", flush=True) #le flush=true est obligatoire quand on est dans un sous processus
 
# protect the entry point
if __name__ == '__main__':
    syn0, global_word_count = init_shared_variables()
    # create and configure the process pool
    pool = Pool(processes=2, initializer=initialize_worker, initargs=(syn0, global_word_count))
    # issue tasks to the process pool
    #for _ in range(2):
     #   pool.apply_async(task)
    pool.map_async(task, range(2))
    # close the process pool
    pool.close()
    # wait for all tasks to complete
    pool.join()

    print(f"le tableau est devenu {'|'.join([str(t) for t in syn0_tmp])}")
    print(f"le global word count vaut {global_word_count.value}")
