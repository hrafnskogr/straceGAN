#! /usr/bin/python3
import os, sys, argparse, time
import tempfile
import subprocess
from multiprocessing import *
from xml.dom.minidom import parseString
from shutil import copy, rmtree, copytree
from os import path
import threading

_args = None
_permissions = []
_pkg_name = ""
_main_activity = ""
_adb_tid = 0

def main():
    global _args
    global _pkg_name
    global _main_activity
    global _adb_tid

    args_inquisitor()

    dirs = []

    dirs = os.listdir(_args.datapath)
    path = format_path(_args.datapath)

    if(_args.started is False):
        inform("Starting emulator {}".format(_args.vdname))
        emul_admin('start', _args.vdname)
    else:
        inform("Emulator already started")

    inform("Getting ADB transport id")
    get_adb_transport_id()
    inform("ADB Transport id: {}".format(_adb_tid))

    failed = {}         # not used yet, will see

    cont = False
    if(_args.continued is not None):
        cont = True

    with open("apk_install_results", 'a', buffering=1) as report:        # can add buffering=1 to flush each line one by one to the file
        for directory in dirs:
            if(os.path.isdir(path + directory)):
                tmpPath = path+directory

                apks = os.listdir(tmpPath)
                for apk in apks:
                    if(apk[-3:] == "apk"):
                        if(cont):
                            if(apk == _args.continued):
                                inform("Skipping last {}".format(apk))
                                cont = False
                                continue
                            else:
                                inform("Skipping {}".format(apk))
                                continue

                        fullpath = format_path(tmpPath) + apk
                        tmpPath = format_path(tmpPath)
                        # ---------------- start of processing logic

                        hasManifest = apkHasManifest(fullpath)
                        pkname = ""

                        msg = ""

                        if(hasManifest):
                            inform("Reading package name from {}...".format(apk))
                            aapt = extractPackageName(fullpath)
                            err = aapt.stderr.decode()

                            decoded = False
                            decodeFail = False

                            if( ("no AndroidManifest.xml found" in err) or ("ERROR" in err) ):
                                inform("aapt failed, trying AndroidManifest.xml...")
                                apktool = apkDecode(tmpPath, apk)

                                print(apktool.stderr.decode())
                                print(apktool.stdout.decode())

                                if("Exception" not in apktool.stderr.decode() and "Exception" not in apktool.stdout.decode()):
                                    decoded = True
                                    pkname = extractPackageNameFromManifest(tmpPath+"decoded/AndroidManifest.xml")
                                else:
                                    decodeFail = True
                                    pkname = "not found"
                            else:
                                pkname = aapt.stdout.decode().split('\n')[0].split(' ')[1][6:-1]

                            inform("PackageName found: {}".format(pkname))

                            inform("Pushing APK for installation: {}".format(fullpath))

                            if(_args.started):
                                process = emul_device('install', fullpath)
                                status = process.stderr.decode()
                            else:
                                process = adb(['install', '-r', fullpath])
                                status = process.stderr.decode()
                                print(status)

                            if( ("Failure" in status) and (not decodeFail)):
                                error = status.split("\n")[4].split(" ")[1]
                                print(error)
                                failed[apk] = error

                                msg = "{} - {}".format(apk, error.strip())

                                if("[INSTALL_FAILED_INVALID_APK]" in error):
                                    inform("Rebuilding apk...")
                                    rebuildApk(tmpPath, apk, decoded)
                                    # retry
                                    process = emul_device('install', tmpPath+"decoded/dist/"+apk)
                                    status = process.stderr.decode()
                                    if("Failure" not in status):
                                        inform("Rebuilding success")
                                        msg += " --- FIXED!"
                                        saveNewApk(tmpPath, tmpPath+"decoded/dist/"+apk, apk)

                                    cleanRebuild(tmpPath+"decoded/")
                            elif("Failure" in status):
                                msg = "{} - {}".format(apk, status.strip())
                            else: # success
                                #print(status)
                                inform("Install success")
                        else:
                            msg = "{} - NOT AN APK!! (no manifest)".format(apk)

                        if(pkname != ""):
                            inform("Uninstalling {}".format(pkname))
                            adb(['uninstall', pkname])

                        inform("FINAL WARNING: {}\n\n".format(msg))

                        report.write(msg+"\n")

    inform("Stoping emulator")

    if(_args.started is False ):
        emul_admin('stop', _args.vdname)
        inform("...cleaning...")
        vd_path = format_path(_args.vdpath) + _args.vdname + "/"
        master = vd_path + 'master'
        rmtree(vd_path + 'Snapshots')
        copytree(vd_path + 'master', vd_path + 'Snapshots')
    else:
        inform("Emulator was started, doing nothing...")

    inform("...DONE!")

    sys.exit(0)     # everything ended well, give ok result to polling parent process

def apkHasManifest(apkPath):
    z = subprocess.run(["7z", "l", apkPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lst = z.stdout.decode()

    if("AndroidManifest" in lst):
        return True
    else:
        return False

def extractPackageNameFromManifest(mpath):
    data = ''
    with open(mpath, 'r') as f:
        data = f.read()
    dom = parseString(data)

    pkg = dom.getElementsByTagName('manifest')
    return pkg[0].getAttribute('package')

def extractPackageName(apkpath):
    # might need to add full path of aapt or ensure that it's in your path
    aapt = subprocess.run(["aapt", "dump", "badging", apkpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return aapt

def saveNewApk(path, newApkPath, apk):
    copy(path+apk, path+apk[:-3]+"bak")
    copy(newApkPath, path+apk)

def cleanRebuild(decodedPath):
    rmtree(decodedPath)

def rebuildApk(path, apk, decoded):
    apkDecode(path, apk, decoded)
    apkBuild(path, apk)
    apkSign(path, apk)

def apkDecode(path, apk, decoded=False):
    if(not decoded):
        apktool = subprocess.run(["apktool", "d", "-f", path+apk, "-o", path+"decoded"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return apktool
        #inform("Apktool exit code: {}".format(apktool.returncode))

def apkBuild(path, apk):
    apktool = subprocess.run(["apktool", "b", path+"decoded"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #inform("Apktool exit code: {}".format(apktool.returncode))

def apkSign(path, apk):
    jarsign = subprocess.run(["jarsigner", "-verbose", "-sigalg", "SHA1withRSA", "-digestalg", "SHA1", "-keystore", "/home/hrafn/malstore.keystore", "-storepass", "malpass", path+"decoded/dist/"+apk, "malkey"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #inform("Jarsigner exit code: {}".format(jarsign.returncode))

def get_adb_transport_id():
    global _args
    global _adb_tid

    genlist = subprocess.run(['/opt/genymobile/genymotion/gmtool', 'admin', 'list'], stdout=subprocess.PIPE)
    genlist = genlist.stdout.decode().split('\n')

    adb_ip = ""

    for line in genlist:
        if(_args.vdname in line):
            adb_ip = line.split('|')[1].strip()
            break

    adblist = subprocess.run(['adb', 'devices', '-l'], stdout=subprocess.PIPE)
    adblist = adblist.stdout.decode().split('\n')

    for line in adblist:
        if(adb_ip in line):
            _adb_tid = line[line.find("transport_id:") + len("transport_id:"):]

def save_strace_file(path):
    adb(['pull', '/sdcard/straces.txt', path])

def run_monkey(nevents):
    global _pkg_name
    adb(['shell', 'monkey', '-p', _pkg_name, nevents])

def start_strace():
    global _adb_tid

    time.sleep(1)
    adb(['root'])
    adb(['shell', '"set enforce 0"'])
    adb(['shell', 'am', 'startservice', '-n', '"com.hrafnskogr.stracerv/.stracerv"'])
    time.sleep(5)

def stop_strace():
    adb(['shell', 'am', 'stopservice', '-n', '"com.hrafnskogr.stracerv/.stracerv"'])

def adb(cmd, shell=False):
    global _adb_tid
    core = ['adb', '-t', _adb_tid]
    adb = subprocess.run(core + cmd, shell=shell)
    return adb

def emul_device(cmd, cmdparam):
    global _args
    proc = subprocess.run(['/opt/genymobile/genymotion/gmtool', 'device', '-n', _args.vdname, cmd, cmdparam], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc

def emul_admin(cmd, param):
    proc = subprocess.run(['/opt/genymobile/genymotion/gmtool', 'admin', cmd, param])
    return proc.returncode

def format_path(path):
    if(path[-1:] == "/"):
        return path
    else:
        return path + "/"

def analyse_manifest(path):
    global _permissions
    global _pkg_name
    global _main_activity

    data = ''
    with open(path, 'r') as f:
        data = f.read()
    dom = parseString(data)
    nodes = dom.getElementsByTagName('uses-permission')

    for node in nodes:
        _permissions.append(node.getAttribute('android:name'))

    pkg = dom.getElementsByTagName('manifest')
    _pkg_name = pkg[0].getAttribute('package')

    activities = dom.getElementsByTagName("activity")
    for activity in activities:
        intents = activity.getElementsByTagName("intent-filter")
        for intent in intents:
            actions = intent.getElementsByTagName("action")
            for action in actions:
                if("MAIN" in action.getAttribute("android:name")):
                    _main_activity = activity.getAttribute("android:name")
                    break

def inform(msg):
    print("\033[94m[-]\033[0m {}".format(msg))

def args_inquisitor():
    global _args
    parser = argparse.ArgumentParser(description="Test APK installation, try to resign the ones that can be - and log results")
    parser.add_argument('datapath', help='original apk file to process')
    parser.add_argument('-n', '--vdname', help="virtual device name to use")
    parser.add_argument('-p', '--vdpath', help="genymotion virtual device path")
    parser.add_argument('-c', '--continued', help="continue at this apk")
    parser.add_argument('-s', '--started', action='store_true')
    _args = parser.parse_args()


if(__name__ == "__main__"):
    main()
