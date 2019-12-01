#! /usr/bin/python
import os, argparse, sys, re
import glob

_args = None
_re_syscall = '^.*\('
_syscalls = {}  # debug

def main():
    global _args
    global _syscalls

    args_inquisitor()
    if(not _args.no_preprocess):
        process_files(format_path(_args.dataset))
    else:
        print("Skipping preprocess of files")

    if(_args.build):
        build_final()

def build_final():
    print("Building final dataset file...")

    # rewritten and not tested
    files = glob.glob(_args.output)
    with open(format_path(_args.output) + "dataset.build.txt", 'w') as build:
        for file in files:
            with open(file, 'r') as inputfile:
                for line in inputfile:
                    build.writeline(line)

def process_files(input_path):
    global _args
    files = glob.glob(input_path + "*.strace.txt")
    nfiles = len(files)

    for i,file in enumerate(files):
        if(not _args.test):
            print(f"{i}/{nfiles}")
        data = refactor_file(file)
        outpath = format_path(_args.output) + file[-43:]    # outpath + filename only which is md5.strace.txt (43 chars)
        if(not _args.test):
            write_data(outpath, data)

def refactor_file(file):
    global _args
    global _re_syscall
    global _syscalls
    new_file = []

    with open(file, 'r') as f:
        ite = 0
        for line in f:
            l = line.split(' ')
            newline = []
            for part in l:
                if(part != ''):
                    newline.append(part)
            time, syscall = newline[1], newline[2]

            if(syscall[0] == "<"):                                  # 1811       0.000016 <... futex resumed> ) = 0 <0.000045>
                syscall = f"{newline[3]} {newline[4][:-1]}"         # syscall = "futex resumed"
            elif(syscall.startswith("---")):                        # 1740       0.000017 --- SIGSEGV {si_signo=SIGSEGV, si_code=SEGV_MAPERR, si_addr=0} ---
                syscall = newline[3].lower()                        # syscall = "sigsev"
            elif(syscall.startswith("+++")):                        # 1718       0.000886 +++ killed by SIGKILL +++
                syscall = newline[5].lower()                        # syscall = "sigkill"
            elif(syscall.startswith("???")):
                continue
            else:                                                   # 1761       0.000322 write(74, "W", 1 <unfinished ...>
                regex = re.search(_re_syscall, syscall)             # .
                if(regex is not None):                              # .
                    s = regex.start()                               # .
                    e = regex.end()                                 # .
                    syscall = syscall[s:e-1]                        # syscall = "write"
                else:
                    if(not _args.test):
                        print(f"DEBUG:\n{file}\n{ite} - {line}")
                        print("Could not parse file, incorrect format. \n\n\tplease fix\t\n\n")
                        sys.exit()
                    else:
                        print(f"{file}")


            if(_args.minimal_mode):
                new_file.append(f"{syscall}")
            elif(_args.short_mode):
                new_file.append(f"{time} {syscall}")
            elif(_args.medium_mode):
                print("MEDIUM MODE NOT IMPLEMENTED YET")
                pass
            ite += 1

    return " ".join(new_file)

def write_data(outpath, data):
    print(outpath)

    with open(outpath, "w") as out:
        out.write(data)

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
    parser = argparse.ArgumentParser(description="Refactor extracted strace from apk sample to file ready to be used for deep learning")
    parser.add_argument('dataset', help='Path of the dataset')
    parser.add_argument('-o', '--output', help='output path of the refactored dataset')
    parser.add_argument('-s', '--sort', action='store_true', default=False, help='sort the refactored syscall by thread id, not implemented')
    parser.add_argument('-b', '--build', action='store_true', default=False, help='build a single dataset file ready for tokenization (concatenate all files)')
    parser.add_argument('-n', '--no_preprocess', action='store_true', default=False, help='Skip file refactoring')
    parser.add_argument('--test', action='store_true', default=False, help="test only the parsing of the files")

    mutex = parser.add_mutually_exclusive_group()
    mutex.add_argument('-0', '--minimal_mode', action='store_true', default=False, help="keep only syscalls")
    mutex.add_argument('-1', '--short_mode', action='store_true', default=False, help='keep only time and syscall')
    mutex.add_argument('-2', '--medium_mode', action='store_true', default=False, help='keep only time, syscall, and shortened argument (not implemented)')

    _args = parser.parse_args()


if(__name__ == "__main__"):
    if(os.name == "nt"):
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    main()
