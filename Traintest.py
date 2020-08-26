from sklearn.utils import shuffle
import os,argparse, h5py
import numpy as np
from zipfile import BadZipFile
from Helpers import load_pickle

def load_npz(path):

    '''Load data, labels and domain names for all files in the target folder'''

    data=[]
    labels=[]
    files = shuffle([file for file in os.listdir(path) if file[-4:]=='.npz'],random_state=42)[:1000]
    print('Loading {} files from {}\n'.format(len(files), path))
    n = 1
    for file in files:
        if n % 1000 == 0:
            print('{}/{}'.format(n, len(files)))

        upload = np.load(path + '/' + file, allow_pickle=True)
        data.append(upload['img'])
        labels.append(upload['CATH'])
        n += 1
    print('Complete')

    return data, labels,[file[:-4] for file in files]

def save_data(savefile, data, labels,domains):

    '''Save data to a single .h5 file as an intermediate step to generating training and test splits'''

    print('Saving data to ',savefile)
    with h5py.File(savefile, 'w') as hf:
        hf.create_dataset('data', data=np.asarray(data))
        hf.create_dataset('cath_label', data=np.asarray(labels, dtype='S'))
        hf.create_dataset('domains', data=np.asarray(domains, dtype='S'))
        hf.close()
    print('Save complete')

def load_data(file):

    '''Load data from .h5 file '''

    print('Loading from %s:'%(file))
    with h5py.File(file, 'r') as hf:
        data = np.asarray(hf['data'])
        cath = np.asarray(hf['cath_label'])
        cath = [[x.decode('UTF-8') for x in i.tolist()] for i in cath]
        domains = np.asarray(hf['domains'])
        domains = [x.decode('UTF-8') for x in domains]
    print('Load complete')

    return data, cath, domains

def one_hot(labels, onehotdict):

    '''One-hot encode labels using a lookup dictionary generated from the CATH v4.2.0 domain list,
    available at http://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt'''

    print('One hot encoding labels:')
    names = ['C', 'A', 'T', 'H']
    out = []
    nums = []
    for l in range(len(names)):                            # For each of the C, A, T and H tasks
        label_subset = [item[l] for item in labels]
        num = len(onehotdict[names[l]].keys())
        nums.append(num)                                    # Record the number of categories per class
        onehot = np.zeros((len(label_subset), num))         # Create an output matrix of width = #classes,
                                                            # length = # instances
        for i in range(len(label_subset)):
            try:
                onehot[i][onehotdict[names[l]][label_subset[i]]] = 1  # Set the index corresponding to that label to 1
            except KeyError:
                onehot[i][onehotdict[names[l]]['<UNK>']] = 1            # To account for out of vocabulary instances
        out.append(onehot)

    print('%s C, %s A, %s T and %s H labels excluding <UNK>' % (nums[0] - 1, nums[1] - 1,
                                                                nums[2] - 1, nums[3] - 1))
    return out, nums


def train_test(data, labels, domains, nums, test_ratio=False):

    '''Split dataset into training and test sets'''

    C = labels[0]
    A = labels[1]
    T = labels[2]
    H = labels[3]

    if not test_ratio:
        test_ratio = 0.1
    elif (test_ratio>1 )|(test_ratio<0):
        print('Error: please enter a value between 0 and 1')

    print('Splitting into {:.0f}% train, {:.0f}% test'.format((1 -test_ratio)*100,test_ratio*100))

    start = int(test_ratio * len(data))

    # Training data
    if test_ratio ==1:
        X_train=[]
        C_train=[]
        A_train=[]
        T_train=[]
        H_train=[]
        trainlist = []
    else:
        X_train = np.array(data[start:])
        C_train = C[start:]
        A_train = A[start:]
        T_train = T[start:]
        H_train = H[start:]
        trainlist=np.asarray(domains[start:],dtype='S')

    # Test data

    X_test = np.array(data[:start])
    C_test = C[:start]
    A_test = A[:start]
    T_test = T[:start]
    H_test = H[:start]
    testlist=np.asarray(domains[:start],dtype='S')

    print('Data split into {} train and {} test set instances'.format(len(X_train), len(X_test)))


    # Load to dictionary

    split = {}
    split['X_train'] = X_train
    split['C_train'] = C_train
    split['A_train'] = A_train
    split['T_train'] = T_train
    split['H_train'] = H_train
    split['Train_domains']=trainlist
    split['X_test'] = X_test
    split['C_test'] = C_test
    split['A_test'] = A_test
    split['T_test'] = T_test
    split['H_test'] = H_test
    split['Test_domains'] = testlist
    split['Classes'] = nums

    return split

def save_traintest(filepath, traintest):

    '''Save train test split to .h5 file'''

    if filepath[-3:]!= '.h5':
        filepath+='.h5'
    print('Saving to %s:'%(filepath))
    out = h5py.File(filepath, 'w')
    for k in traintest.keys():
        print(k)
        out.create_dataset(k, data=traintest[k])
    out.close()
    print('Save complete')


def load_traintest(filepath):

    print('Loading data')
    if filepath[-3:] != '.h5':
        filepath += '.h5'

    h5f = h5py.File(filepath, 'r')

    X_train, C_train,A_train,T_train,H_train, _,X_test,C_test,A_test,T_test,H_test,_,nums = [np.asarray(h5f[x]) for x in [
        'X_train', 'C_train', 'A_train', 'T_train', 'H_train',
        'Train_domains', 'X_test', 'C_test', 'A_test', 'T_test',
        'H_test', 'Test_domains', 'Classes']]
    return (X_train, (C_train, A_train, T_train, H_train), X_test, (C_test, A_test, T_test, H_test)), nums

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generate train and test files from preprocessed inputs')

    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help='The input folder for generating training and test sets')
    parser.add_argument('-tr', '--test_ratio', type=float, required=False,
                        help='The proportion of the training set to set aside for testing')


    args = parser.parse_args()

    # data,labels,domains=load_npz(args.input_folder)
    dir=args.input_folder.split('/')
    outpath=os.path.join(*dir[:-1])+'/%s%s_ALL.h5'%(dir[-1],dir[-3].upper())
    # save_data(outpath,data,labels,domains)
    # data,cath,domains = load_data(outpath)
    onehotdict=load_pickle('CATHonehot.pickle')
    # labels, nums = one_hot(cath,onehotdict)
    # traintest = train_test(data, labels, domains, nums, args.test_ratio)
    savepath= os.path.join(*dir[:-1])+'/%s%s_TRAINTEST.h5'%(dir[-1],dir[-3].upper())
    # save_traintest(savepath,traintest)
    data = load_traintest(savepath)
    print(data[1])
