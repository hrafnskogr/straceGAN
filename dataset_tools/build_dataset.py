import sys, os
import argparse
import pickle
import glob
from os import path

_args = None

def tokenize_dataset():
    global _args

    one_name, one_path, one_meta = create_dirs(_args.out_one)
    two_name, two_path, two_meta = create_dirs(_args.out_two)

    tok2txt, txt2tok = vocabulary_mode()

    # save metadata
    print("Saving metadata")
    meta = {}
    meta["name"] = one_name
    meta["tok2txt"] = tok2txt
    meta["txt2tok"] = txt2tok
    save_metadata(meta, one_meta, one_name)
    meta["name"] = two_name
    save_metadata(meta, two_meta, two_name)

    data_one_files = glob.glob(format_path(_args.data_one) + "*.strace.txt")
    data_two_files = glob.glob(format_path(_args.data_two) + "*.strace.txt")


    print("preprocessing files")

    tokenize(data_one_files, one_path, txt2tok)
    tokenize(data_two_files, two_path, txt2tok)

    print("\nDone.\n")

def tokenize(files, out_path, text_to_token):
    global _args

    slen = _args.seqLen
    ite = 1
    for f in files:
        print(f"[{ite}/{len(files)}]")
        with open(f, 'r') as input_sample:
            path = f.split(".")
            fname = path[len(path)-3][-32:] #f.split(".")[0]
            with open(format_path(out_path) + fname + ".str", 'w') as output_sample:
                intext = input_sample.read().split()
                outtext = [text_to_token[token] for token in intext]
                output_sample.write(" ".join(str(int_token) for int_token in outtext[:slen+1]))       # truncate input file to only the required sequence lenght
        ite += 1

def build_vocabulary():
    global _args

    tok2txt, txt2tok = vocabulary_mode()

    data = {}
    data["size"] = len(tok2txt)
    data["tok2txt"] = tok2txt
    data["txt2tok"] = txt2tok

    with open(format_path(_args.vocab_path) + "vocab.pkl", "wb") as vocabfile:
        pickle.dump(data, vocabfile)

    print(tok2txt)
    print(txt2tok)
    print(len(tok2txt))

def vocabulary_mode():
    global _args
    token_to_text = {}
    text_to_token = {}

    print("Building vocabulary dictionary")

    data_one_files = glob.glob(format_path(_args.data_one) + "*.strace.txt")
    data_two_files = glob.glob(format_path(_args.data_two) + "*.strace.txt")

    all_files = data_one_files + data_two_files
    words = {}
    for i, f in enumerate(all_files):
        print(f"[{i}/{len(all_files)}] - Scanning {f}")
        with open(f, 'r') as sample:
            text = sample.read().split()
            for token in text:
                if(token in words):
                    words[token] += 1
                else:
                    words[token] = 1

    for key, value in enumerate(words):
        token_to_text[key] = value

    for key, value in enumerate(words):
        text_to_token[value] = key

    return token_to_text, text_to_token

def save_metadata(metadata, path, name):
    with open(format_path(path) + name + ".meta.pkl", 'wb') as metapath:
        pickle.dump(metadata, metapath)

def create_dirs(base_path):
    global _args

    dataset_name = ""
    dataset_path = ""
    metadata_path = ""

    if(os.name == 'nt'):
        if(base_path[-1:] == "\\"):
            dataset_path = base_path[:-1]
        else:
            dataset_path = base_path

        dataset_name = path.basename(dataset_path)
        metadata_path = dataset_path + "\\meta_" + dataset_name
    else:
        print("not implemented yet")
        pass

    if not os.path.exists(dataset_path):
        os.makedirs(metadata_path)

    print(f"Dataset Name: {dataset_name}\nSamples Path: {dataset_path}\nMetadata Path: {metadata_path}")
    ans = input("Continue? [y/n]: ").lower()

    while(ans not in ['y', 'n']):
        print(f"Dataset Name: {dataset_name}\nSamples Path: {dataset_path}\nMetadata Path: {metadata_path}")
        ans = input("Continue? [y/n]: ").lower()

    if(ans == 'n'):
        print("Aborting...")
        sys.exit(0)

    return dataset_name, dataset_path, metadata_path

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

def args_inquisitor():
    global _args
    parser = argparse.ArgumentParser(description="Convert extracted strace from apk sample to usable tokenized dataset. \nWill use a cleanware and malware dataset to build a common vocabulary file.\nOutputs files in a tokenized form, and a metadata pickle file storing vocabulary dictionaries (txt to tokens and inverse), and size of vocabulary.\nThe vocabulary file is also saved on its own for separate use.")
    parser.add_argument('--data_one', help='first dataset for vocabulary mode')
    parser.add_argument('--data_two', help='second dataset for vocabulary mode')
    parser.add_argument('--out_one', help='output path of the refactored dataset, will be used as name for the dataset')
    parser.add_argument('--out_two', help='output path of the refactored dataset, will be used as name for the dataset')
    parser.add_argument('-p', '--vocab_path', help="vocabulary save path")
    parser.add_argument('-l', '--seqLen', type=int, help='Length of sequence to be used. Files will be truncated to this size + 1')
    parser.add_argument('-v', '--vocab', action='store_true', default=False, help='Build vocabulary from two datasets')

    _args = parser.parse_args()

if(__name__ == '__main__'):
    args_inquisitor()

    if(_args.vocab):
        build_vocabulary()
        sys.exit()

    tokenize_dataset()
