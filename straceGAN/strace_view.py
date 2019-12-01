import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

_args = None
_dir = "/"



if(os.name == 'nt'):
    _dir = "\\"



def format_path(path):
    if(os.name == 'nt'):
        if(path[-1:] == "\\"):
            return path
        else:
            return path + "\\"
    else:
        if(path[-1:] == "/"):
            return path
        else:
            return path + "/"

def load_vocab(pfile):
    vocab_data = {}
    with open(pfile, 'rb') as vfile:
        vocab_data = pickle.load(vfile)
    tok2txt = vocab_data['tok2txt']
    txt2tok = vocab_data['txt2tok']
    vocab_size = vocab_data['size']

    return txt2tok, vocab_size

def args_inquisitor():
    global _args
    parser = argparse.ArgumentParser(description="Transforms a dataset into an image, for data visualization")
    parser.add_argument('-d', '--dataset', help='the dataset to analyze')
    parser.add_argument('-v', '--vocab_file', help='vocabulary file data from a pickle file', type=str, default=None)
    parser.add_argument('-o', '--output', help='where to save the image, not implemented yet, use the pyplot UI instead.')
    parser.add_argument('-i', '--int', action='store_true', help='input files are already tokenized', default=False)

    _args = parser.parse_args()



if(__name__ == '__main__'):
    args_inquisitor()

    txt2tok,svocab = load_vocab(_args.vocab_file)
    files = glob.glob(f"{format_path(_args.dataset)}*.str")

    x = 0
    y = len(files)

    with open(files[0], 'r') as f:
        x = len(f.read().split())

    print(files[0])

    data = np.zeros((x, y))

    print(np.shape(data))

    for i,sample in enumerate(files):
        with open(sample, 'r') as s:
            if(not _args.int):
                colours =  [txt2tok[key] for key in s.read().split()]
            else:
                colours = s.read().split()
            if(len(colours) < x ):
                print(f"skipping sample {sample} of len {len(colours)}")
                continue
            data[:,i] = colours

    plt.imshow(data, interpolation='none', vmin=0, vmax=svocab)
    plt.show()
