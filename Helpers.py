import multiprocessing as mp
import pickle
from csv import writer

def parallelise(function, input_data, args):
    '''Parallelise function over available CPU cores.
    Adapted from https://tutorialedge.net/python/python-multiprocessing-tutorial/'''

    num_jobs = mp.cpu_count()

    print('Parallelising across {} cores'.format(num_jobs))
    chunk = int(len(input_data) / num_jobs)  # Divide the array into equal chunks
    chunked = [input_data[(i * chunk):(i + 1) * chunk] for i in range(num_jobs - 1)]
    ext = [input_data[(num_jobs - 1) * chunk:]]
    chunked.extend(ext)
    processes = [mp.Process(target=function, args=(chunked[i], *args)) for i in range(len(chunked))]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

def load_pickle(file):
    with open(file, 'rb') as handle:
        out = pickle.load(handle)
        return out

def writeline(csv,line):
    with open(csv, 'a') as c:
        write=writer(c)
        write.writerow(line)
