#! /usr/bin/python3
import zipfile
import sys
import hashlib
import tempfile
import os
import subprocess
import csv
from shutil import copy
from os import walk

_workfolder = sys.argv[1]
_outfolder = sys.argv[2]
_curzip = None
_files = None
_file_error = []
_noapk_error = []
_proto_csv = []

def init():
    global _workfolder
    global _outfolder

    # path sanitization
    if(_workfolder[-1:] != "/"):
        _workfolder += "/"

    if(_outfolder[-1:] != "/"):
        _outfolder += "/"

    # get all files and corresponding paths
    (_, _, files) = next(walk(_workfolder))

    ret = []

    for f in files:
        ret.append(_workfolder + f)

    return ret

def extract(f, tmp_path):
    global _tmp
    global _file_error

    pw = ['#', '#', '#']        # set passwords

    for p in pw:
        print("Trying {}".format(p))
        sub = subprocess.Popen(['7z', 'e', '-y', '-p{}'.format(p), '-o{}'.format(tmp_path), f], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ret = sub.communicate()[0]
        if(sub.returncode == 0):
            print("\tpass is {}".format(p))
            return

    _file_error.append(f)

def md5hash(path):
    sample = ""
    files = []
    for f in os.listdir(path):
        fullpath = path+"/"+f
        if(not os.path.isdir(fullpath)):
            if f.endswith(".apk") or f.endswith(".APK") or is_apk(path+"/"+f):
                print("\t [#] hashing {}".format(f))
                h = hashlib.md5()
                with open(path+"/"+f, "rb") as s:
                    h = hashlib.md5()
                    while True:
                        buf = s.read(4096)
                        if not buf:
                            break
                        h.update(buf)
                files.append((h.hexdigest(), fullpath))
    return files

def is_apk(f):
    # use magic number to determine if file could possibly be an apk
    apkmagic = b'\x50\x4b\x03\x04'
    with open(f, 'rb') as unk:
            magic = unk.read(4)
            if(magic == apkmagic):
                return True
            else:
                return False

def create_sample_dir(outfolder, h):
    d = outfolder + h
    try:
        os.mkdir(d)
    except:
        pass
    #except Exception as e:
    #    print(e)

    return d

def store_sample(src, dst, h):
    dst += "/" + h + ".apk"
    print(dst)
    copy(src, dst)

def write_csv():
    global _outfolder
    global _proto_csv

    with open(_outfolder + "sampledb.csv", mode="w") as sampledb:
        dbwriter = csv.writer(sampledb, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        dbwriter.writerow(['original_package', 'hash'])

        for sample in _proto_csv:
            dbwriter.writerow([sample[0], sample[1]])

def report_errors():
    global _file_error
    global _noapk_error

    print('='*25)
    print("Errors:")
    print("\t [!] Files that were not extracted: [!]")
    for err in _file_error:
        print(err)
    print()
    print("\t [!] No apk extracted from: [!]")
    for no in _noapk_error:
        print(no)

if __name__ == "__main__":
    _files = init()

    h = ""
    sample_path = ""
    storage_path = ""

    for f in _files:
        with tempfile.TemporaryDirectory() as tmp:
            print("Treating: {}".format(f))
            extract(f, tmp)
            files = md5hash(tmp)
            print("")

            if(len(files) == 0):
                print("No apk extracted from {}".format(f))
                _noapk_error.append(f)
                continue

            for ff in files:
                _proto_csv.append((f, ff[0]))
                storage_path = create_sample_dir(_outfolder, ff[0])
                store_sample(ff[1], storage_path, ff[0])

    write_csv()
    report_errors()
