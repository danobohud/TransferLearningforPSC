import os, math, argparse,requests,cv2
import numpy as np
from prody import *
from Helpers import parallelise, load_pickle
from zipfile import BadZipFile

def prepare_directories(atom_selection, root=False):
    '''Create directory tree for source files'''

    pwd = os.getcwd()
    cath, pqr = [os.path.join(pwd, x) for x in ['CATH', 'CATH/PQR']]
    target_folder = os.path.join(cath, atom_selection)
    subdirectories = [os.path.join(target_folder, x) for x in ['raw', 'preprocessed']]
    for directory in [cath, pqr, target_folder, *subdirectories]:
        if not os.path.isdir(directory):
            os.mkdir(directory)
    if not os.path.isdir(os.path.join(pwd, 'dompdb')):
        print('Error: you must have downloaded the S20 and S40 CATH non-redundant set and extracted the files to a folder named "dompdb". See readme for further information.')
    else:
        dompdb = os.path.join(pwd, 'dompdb')
    raw = os.path.join(cath, atom_selection, 'raw')
    return dompdb, pqr, raw

def make_pqr(files, input_folder, output_folder, pdbpqr_path):
    ''' Parse .pdb structure files from CATH non_redundant set'''
    n = 1
    for file in files:
        name = file[:-4]
        if n % 1000 == 0:
            print('----------------\n----------------\n{}/{}'.format(str(n),
                                                                     len(files)))
        input_file = os.path.join(input_folder, file)
        output_file = os.path.join(output_folder, name + '.pqr')
        os.system('python {} {} {} --ff AMBER'.format(pdbpqr_path, input_file, output_file))
        n += 1
    print('Complete')


def get_distmap(atoms):
    '''Compute Euclidean distance between atoms of the input structure'''

    output = np.zeros((len(atoms), len(atoms)))  # Set the distance map proportions
    for x in range(0, len(output)):
        for y in range(0, len(output)):
            if output[x, y] == 0:
                output[x, y] = prody.measure.measure.calcDistance(atoms[x],
                                                                  atoms[y])  # Calculate distances between atoms
                output[y, x] = output[x, y]
    img = output.astype('uint8')

    return (img)


def get_NB(prot, distmat, mincut=-0.1, maxcut=0.1):
    '''Convert structure file and distance matrix into non-bonded
    energy potentials (Van der Waals + Electrostatics). Code provided by Dr Tobias Sikosek following
    the methodology of his 2019 paper https://www.biorxiv.org/content/10.1101/841783v1'''

    n = len(distmat)
    nbmat = np.zeros((n, n))  # Define output matrix
    charges = prot.getCharges()  # Charges for all atoms of the structure
    for i in range(n):
        q_i = charges[i]
        for j in range(n):
            if i < j:
                q_j = charges[j]
                r_ij = distmat[i, j]  # Get distance between atoms

                if r_ij > 0.0:  # and r_ij<=25.6:

                    # https://en.wikipedia.org/wiki/AMBER

                    sig = 12.0  # distance where potential is zero
                    r0_ij = 1.122 * sig  # distance where potential at minimum
                    r6 = math.pow(r0_ij, 6)
                    V_ij = 1.0  # -120.0
                    e_r = 1.0 * math.pi * 4
                    f = 1.0
                    A_ij = 2 * r6 * V_ij
                    B_ij = (A_ij / 2.0) * r6
                    E_LJ = (-A_ij) / math.pow(r_ij, 6) + (B_ij) / math.pow(r_ij, 12)  # Compute Van der Waals potential
                    E_coul = f * (q_i * q_j) / (e_r * r_ij)  # Compute electrostatic energy potential
                    E_nb = E_LJ + E_coul
                    E_nb = min(maxcut, max(mincut, E_nb))

                else:
                    E_nb = 0.0

                nbmat[i, j] = E_nb

                nbmat[j, i] = E_nb

    return nbmat

def get_anm(struct, atomgroup):
    '''Calculate anisotropic network model for atoms in the structure'''

    names = ['ca', 'bb', 'heavy']
    if atomgroup not in names:
        print('Pick one of ', names)
    if atomgroup == 'ca':
        struct = struct.ca
        struct_anm = ANM('struct ca')
    elif atomgroup == 'bb':
        struct = struct.bb
        struct_anm = ANM('struct bb')
    if atomgroup == 'heavy':
        struct = struct.heavy
        struct_anm = ANM('struct heavy')
    struct_anm.buildHessian(struct)
    struct_anm.calcModes(n_modes=len(struct))
    anm = calcCrossCorr(struct_anm)

    return anm

def get_rawfiles(files, input_folder, output_folder, atom_selection):
    n = 1
    e = 0
    for file in files:
        name = file[:-4]
        if n % 1000 == 0:
            print('{}/{}'.format(str(n), len(files)))
        try:
            prot = parsePQR(os.path.join(input_folder, file))
            struct = prot.select(atom_selection)
            distmap = get_distmap(struct)
            nb = get_NB(struct, distmap)
            anm = get_anm(struct, atom_selection)
            out = np.stack([distmap, anm, nb], axis=-1)
            np.savez(os.path.join(output_folder, name), img=out, domain=name, allow_pickle=True)
        except FileNotFoundError:
            e += 1
            continue
        except TypeError:
            e+=1
            continue
        except AttributeError:
            e+=1
            continue
        except ValueError:
            e += 1
            continue
    print('Parsing complete with %s errors' % (e))

def preprocess(files,inpath,outpath,CATHdict,size=False):
    if not size:
        size=255
    f=0
    for i,file in enumerate(files):
        if i%1000==0:
            print('%s/%s'%(i,len(files)))
        try:
            name=file[:7]
            C=CATHdict[name]['C']
            A=CATHdict[name]['A']
            T=CATHdict[name]['T']
            H=CATHdict[name]['H']
            EC=CATHdict[name]['EC']
            upload=np.load(os.path.join(inpath,file+'.npz'))
            dist,anm,nb=[upload['img'][:,:,x] for x in range(3)]
            dist = cv2.resize(dist,dsize=(size,size),interpolation=cv2.INTER_CUBIC)
            dist =((np.clip(dist,0,50)/50)*100).astype('int')
            anm=(np.clip(anm,-1,1)*100).astype('float32')
            anm = cv2.resize(anm,dsize=(size,size),interpolation=cv2.INTER_CUBIC).astype('int')
            nb=(cv2.resize(nb.astype('float32'),dsize=(size,size),interpolation=cv2.INTER_CUBIC)*1000).astype('int')
            img=np.stack([dist,anm,nb],axis=-1)
            np.savez(os.path.join(outpath,name),img=img,CATH=[C,A,T,H],EC=EC,allow_pickle=True)
        except BadZipFile:
            print('Corrupt file: ', name)
            f+=1
            continue
        except FileNotFoundError:
            f+=1
            continue
        except OSError:
            f+=1
            continue
    print('Completed with %s files missing'%(f))


def run_preprocess(atom_selection,size):
    # Create directories
    pwd=os.getcwd()
    CATHdict=load_pickle(pwd+'/CATHdict.pickle')
    domains=[[k for k in CATHdict.keys() if CATHdict[k]['CATEGORY']==x] for x in ['HR','LR','NMR']]
    root=os.path.join(pwd,'CATH',atom_selection)
    raw=os.path.join(root,'raw')
    for i,folder in enumerate(['HR','LR','NMR']):
        print('%s:\n'%(folder))
        outpath = os.path.join(root, 'preprocessed', folder)
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        outfiles=[f[:-4] for f in os.listdir(outpath) if f[-4:]=='.npz']
        files=[f2 for f2 in domains[i] if f2 not in outfiles]
        preprocess(files,*[raw,outpath,CATHdict,size])
    print('Complete')

def run(directories, pdb2pqr_path, atom_selection,size):
    dompdb, pqr, raw = directories
    pqrs = [file[:7] for file in os.listdir(pqr) if file[-4:] == '.pqr']
    files = [file for file in os.listdir(dompdb) if file[-4:] == '.pdb' and file[:7] not in pqrs]
    pqr_args = [dompdb, pqr, pdb2pqr_path]
    print('Processing %s PDB files with PDB2PQR and saving to %s' % (len(files), pqr))
    parallelise(make_pqr, files, pqr_args)
    imgs = [file for file in os.listdir(raw) if file[-4:] == '.npz']
    parse_args = [pqr, raw, atom_selection]
    pqrs = [file for file in os.listdir(pqr) if file[-4:] == '.pqr' and file not in imgs]
    print('Getting representations from %s PQR files and saving to %s' % (len(pqrs), raw))
    parallelise(get_rawfiles, pqrs, parse_args)
    print('Splitting images into HR, LR and NMR sets and reshaping to (%s,%s,3)'%(size,size))
    run_preprocess(atom_selection, size)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process PDB files and produce raw representation files')

    parser.add_argument('-a', '--atom_selection', type=str, required=True,
                        help='The atom selection')
    parser.add_argument('-p', '--pdb2pqr_path', required=True,
                        help='Path to pdb2pqr.py executable')
    parser.add_argument('-d', '--directory', required=False,
                        help='The working directory')
    parser.add_argument('-s', '--size', required=False,
                        help='The size of the processed images')

    args = parser.parse_args()

    atom_selection = args.atom_selection
    valid = ['ca', 'bb', 'heavy']
    if atom_selection not in valid:
        print('Error, please select from ca, bb or heavy')
    directories = prepare_directories(atom_selection, args.directory)
    print('Running')
    if args.size:
        size = args.size
    else:
        size=255
    run(directories, args.pdb2pqr_path, atom_selection,size)


