#! /usr/bin/python3
import os, sys, argparse, time, signal
import tempfile
import subprocess
from multiprocessing import *
from xml.dom.minidom import parseString
from shutil import copy, rmtree, copytree
from os import path

_args = None
_permissions = []
_pkg_name = ""
_main_activity = ""
_adb_tid = -1
_apktool = ""
_strace_proc = None
_app_type = ""
_receivers = {}

def main():
    global _args
    global _pkg_name
    global _main_activity
    global _adb_tid
    global _apktool

    args_inquisitor()

    apk_path,apk_name = path.split(_args.apk_path)

    inform("Creating temp directory")
    with tempfile.TemporaryDirectory() as tmp:                                                      # 1. Make tmp work folder
        inform("Copying APK to temp directory")
        tmp_path = format_path(tmp)                                                                 # 2.
        apk_tmp_path = tmp_path + apk_name                                                          # 2. Copy apk to tmp work folder
        copy(_args.apk_path, apk_tmp_path)                                                          # 2.

        inform("Decoding APK...")
        apktool = subprocess.run([_apktool, "d", apk_tmp_path, "-o", tmp_path+"decoded"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)          # 3. Decode apk
        #print("Apktool exit code: {}".format(apktool.returncode))                                   # 3.

        inform("Analysing APK...")
        analyse_manifest(tmp_path+"decoded/AndroidManifest.xml")                                    # 4. Extract permissions

        inform("Starting emulator {}".format(_args.vdname))
        emul_admin('start', _args.vdname)                                                           # 5/6. Get and start emulator instance

        inform("Getting ADB transport id")
        get_adb_transport_id()
        inform("ADB Transport id: {}".format(_adb_tid))

        inform("Waiting for device...")
        time_waited = 0
        restarted = False
        while(not device_ready()):
            time.sleep(5)
            time_waited += 5
            # should add a failsafe that exits and tries next sample, will retry this one later... or maybe reset process => stop emulator, restart, and retry until it gets there
            print(".", end="")
            sys.stdout.flush()
            if(time_waited > 150 and not restarted):
                inform("Trying to restart the emulator...")
                restarted = True
                restart()
            elif(time_waited > 300 and restarted):
                exit_fail()
        inform("Device ready!")

        # try two install
        inform("Pushing APK for installation")
        if(not install_apk(apk_tmp_path)):
            time.sleep(1)
            if(not install_apk(apk_tmp_path)):
                sys.exit(1)
        #ret = emul_device('install', apk_tmp_path)                                                  # 7. Push apk
        #adb(['install', '-r', apk_tmp_path])
        #print(ret)                              # need to check later                              # 8. Check return code

        # must make sure everything is stopped
        inform("Stopping anything that would have started...")
        forceStop()

        # We assume nothing has started yet, so we hook on zygote for strace
        inform("Starting strace on zygote")
        start_strace()                                                                              # 9. Launch strace on zygote

        # Start whatever the apk is (service, app...)
        inform("Launching APK")                                                                     # 10. Launch apk
        apkStart()
        time.sleep(1.5)
        if(not _args.nofuzz):
            fuzz_dialogs()
            time.sleep(1.5)
            fuzz_dialogs()

        inform("Fuzzing...")
        fuzz()

        inform("Stoping strace process")
        stop_strace()

        inform("Saving strace file")
        strace_path = format_path(_args.strace) + apk_name[:-3] + 'strace.txt'                             # 12. Save/pull strace file
        save_strace_file(strace_path)                                                               # 12.

        inform("Writing permissions to file")
        perm_path = format_path(_args.strace) + apk_name[:-3] + 'perms.txt'
        save_permissions(perm_path)

        stop()

        clean()

    inform("...DONE!")
    sys.exit(0)     # everything ended well, give ok result to polling parent process

def install_apk(apk):
    global _adb_tid
    installproc = subprocess.Popen(['adb', '-t', _adb_tid, 'install', '-r', apk])

    waiter = 0

    while(True):
        status = installproc.poll()

        if(status == 0):
            return True
        elif(waiter > 300):
            if(os.name == "nt"):
                os.kill(installproc.pid, signal.CTRL_C_EVENT)
                os.kill(installproc.pid, signal.CTRL_BREAK_EVENT)
            else:
                installproc.kill()

            return False

        waiter += 1
        time.sleep(1)

    return False

def exit_fail():
    global _args
    inform("### FAILURE ###")
    stop()
    clean()
    sys.exit(1)

def stop():
    global _args
    inform("Stoping emulator")
    emul_admin('stop', _args.vdname)                                                            # 13. Stop emulator instance

def clean():
    global _args

    inform("...cleaning...")
    vd_path = _args.vdpath + '/' + _args.vdname                                                 # 14. Clean / reset emulator
    master = vd_path + '/master'                                                                # 14.
    rmtree(vd_path + '/Snapshots')                                                              # 14.
    copytree(vd_path + '/master', vd_path + '/Snapshots')                                       # 14.

def restart():
    inform("=========== INITIATING RESTART PROCESS ===========")

    stop()

    time.sleep(2)

    inform("Starting emulator {}".format(_args.vdname))
    emul_admin('start', _args.vdname)                                                           # 5/6. Get and start emulator instance

    inform("Getting ADB transport id")
    get_adb_transport_id()
    inform("ADB Transport id: {}".format(_adb_tid))

def device_ready():
    global _adb_tid

    if(_adb_tid == -1):
        get_adb_transport_id()
        if(_adb_tid == -1):
            return False

    status = adb_check(['shell', 'dumpsys', 'input_method'])

    if('mSystemReady=true' in status):
        if('mSystemInteractive=true'):
            return True
    else:
        return False

def fuzz():
    global _app_type
    global _args

    if(_app_type == "app"):
        inform("Waiting for {} seconds...".format(_args.runtime))
        time.sleep(_args.runtime)
        inform("Running monkey...")
        run_monkey(_args.monkey)
    elif(_app_type == "serv"):
        inform("Waiting for {} seconds...".format(_args.runtime))
        time.sleep(_args.runtime)
        inform("Extended wait for background service...")
        time.sleep(_args.runtime * 2)
    elif(_app_type == "recv"):
        inform("Waiting for {} seconds...".format(_args.runtime))

    if(_args.nofuzz):
        return

    inform("Starting intents fuzzing session")
    fuzz_intents()

def fuzz_intents():
    global _receivers
    global _pkg_name

    tot = 0
    for recv in _receivers:
        for action in _receivers[recv]:
            tot += 1

    cur = 0
    for recv in _receivers:
        for action in _receivers[recv]:
            receiver = _pkg_name+"/"+recv
            inform(f"Sending [{cur}/{tot}]| {action}  ->  {receiver} |")
            #adb(['shell', 'am', 'broadcast', '-a', action, _pkg_name+"/"+recv])
            adb_background(['shell', 'am', 'broadcast', '-a', action, receiver])
            time.sleep(2)
            fuzz_dialogs()
            time.sleep(1)
            fuzz_dialogs()  # make it twice, just in case
            time.sleep(1)
            fuzz_dialogs()
            cur += 1

def touch(x, y, comment=""):
    print(f"==============> TAP {x},{y} -- {comment}")
    adb(['shell', 'input', 'tap', str(x), str(y)])

def fuzz_dialogs():
    # 800x1280
    #touch(705, 660, comment = "ok button location")     # touch the "ok" button of the android UI has stopped dialog
    #touch(640, 725, comment = "send button location")     # touch the "send" button of the sms send permission asking dialog
    #touch(572, 934, comment = "grant permission location")     # grant mms permission, some samples ask for it
    # 480x854
    touch(439, 507, comment = "ok button location")     # touch the "ok" button of the android UI has stopped dialog
    touch(410, 860, comment = "Activate button location")
    touch(440, 522, comment = "ok button large dialog location")     # touch the "send" button of the sms send permission asking dialog

def forceStop():
    global _pkg_name
    global _app_type

    if(_app_type == "app"):
        adb(['shell', 'am', 'force-stop', _pkg_name])
    elif(_app_type == "serv"):
        adb(['shell', 'am', 'stopservice', _pkg_name])

def apkStart():
    global _pkg_name
    global _app_type
    global _main_activity

    if(_app_type == "app"):
        adb(['shell', 'am', 'start', _pkg_name+"/"+_main_activity])
    elif(_app_type == "serv"):
        adb(['shell', 'am', 'startservice', _pkg_name+"/"+_main_activity])
    elif(_app_type == "recv"):
        adb(['shell', 'am', 'broadcast', '-n', _pkg_name+"/"+_main_activity])

def get_adb_transport_id():
    global _args
    global _adb_tid

    genlist = subprocess.run(['gmtool', 'admin', 'list'], stdout=subprocess.PIPE)
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
            _adb_tid = _adb_tid.strip()

def save_strace_file(path):
    adb(['pull', '/data/straces.txt', path])

def save_permissions(path):
    global _permissions
    with open(path, 'w') as f:
        for perm in _permissions:
            f.write(perm+'\n')

def run_monkey(nevents, touchOnly=False):
    global _pkg_name
    # '--throttle', '20',
    adb(['shell', 'monkey',
                '--ignore-crashes',
                '--ignore-timeouts',
                '--pct-appswitch', '0',
                '--pct-trackball', '0',
                '--pct-nav', '0',
                '-p', _pkg_name, nevents])

def start_strace():
    global _adb_tid
    global _strace_proc

    # old technique, but services doesn't seem to be able to catch all threads
    #time.sleep(1)
    #adb(['shell', 'am', 'startservice', '-n', '"com.hrafnskogr.stracerv/.stracerv"'])
    #time.sleep(1)

    adb(['root'])
    adb(['shell', "setenforce", "0"])
    zygote_pid = adb_check(['shell', 'pgrep', 'zygote']).strip()

    _strace_proc = subprocess.Popen(['adb', '-t', _adb_tid, 'shell', 'strace', '-p', zygote_pid, '-f', '-tt', '-T', '-r', '-s', '128', '-qq', '-o', '/data/straces.txt'])

def stop_strace():
    global _strace_proc
    # old technique
    # adb(['shell', 'am', 'stopservice', '-n', '"com.hrafnskogr.stracerv/.stracerv"'])
    if(os.name == "nt"):
        # os.kill(_strace_proc.pid, signal.CTRL_C_EVENT)
        #os.kill(_strace_proc.pid, signal.CTRL_)
        _strace_proc.kill()
    else:
        os.kill(_strace_proc.pid, signal.SIGINT)

def adb_background(cmd, shell=False):
    global _adb_tid
    core = ['adb', '-t', _adb_tid]
    subprocess.Popen(core+cmd, shell=shell)

def adb_check(cmd, shell=False):
    global _adb_tid
    core = ['adb', '-t', _adb_tid]
    #inform("Running {}".format(core + cmd))
    adb = subprocess.run(core + cmd, shell=shell, stdout=subprocess.PIPE).stdout.decode()
    return adb

def adb(cmd, shell=False):
    global _adb_tid
    core = ['adb', '-t', _adb_tid]
    subprocess.run(core+cmd, shell=shell)

def emul_device(cmd, cmdparam):
    global _args
    proc = subprocess.run(['gmtool', 'device', '-n', _args.vdname, cmd, cmdparam])
    return proc.returncode

def emul_admin(cmd, param):
    proc = subprocess.run(['gmtool', 'admin', cmd, param])
    return proc.returncode

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

def analyse_manifest(path):
    global _permissions
    global _pkg_name
    global _main_activity
    global _app_type
    global _receivers

    data = ''
    try:
        with open(path, 'r', encoding="utf-8", errors='replace') as f:
            data = f.read()
    except FileNotFoundError:
        inform("-------  FILE NOT FOUND  -------")
        inform("------- WRONG APK FORMAT -------")
        sys.exit(1)
    dom = parseString(data)

    # get permissions | todo: export them properly
    nodes = dom.getElementsByTagName('uses-permission')
    for node in nodes:
        _permissions.append(node.getAttribute('android:name'))

    pkg = dom.getElementsByTagName('manifest')
    _pkg_name = pkg[0].getAttribute('package')

    inform("Package is {}".format(_pkg_name))

    # determine what is launchable
    # normal people will use an activity xml node to describe their launchable component
    _app_type = "app"
    _main_activity = manifest_find_main_activity(dom, "activity")
    if(_main_activity == ""):
        # sneaky people will alias it, to avoid automation solely based on activity xml nodes
        _main_activity = manifest_find_main_activity(dom, "activity-alias")
    elif(_main_activity == ""):
        # sneakier sneaky people will craft a service to hide a bit deeper in the Android system
        _app_type = "serv"
        _main_activity = manifest_find_service(dom)
    elif(_main_activity == ""):
        # maybe it's a widget, so we need to get the receiver to try to launch something...
        _app_type = "recv"
        _main_activity = manifest_find_receiver(dom)

    inform("Analyse results: App type is {} | potential EP is {}".format(_app_type, _main_activity))

    # make a list of everything the apps listen for and who the receiver is
    receivers = dom.getElementsByTagName("receiver")
    for receiver in receivers:
        ints = []
        recv = receiver.getAttribute("android:name")
        intents = dom.getElementsByTagName("intent-filter")
        for intent in intents:
                actions = intent.getElementsByTagName("action")
                for action in actions:
                    if( ("android.intent.action.MAIN" in action.getAttribute("android:name")) or
                        ("android.intent.action.LAUNCHER" in action.getAttribute("android:name"))):
                        continue
                    else:
                        ints.append(action.getAttribute("android:name"))
        _receivers[recv] = ints

def manifest_find_receiver(xmlNode):
    app = xmlNode.getElementsByTagName("application")[0]
    receivers = app.getElementsByTagName("receiver")
    return receivers[0].getAttribute("android:name")

def manifest_find_service(xmlNode):
    app = xmlNode.getElementsByTagName("application")[0]
    services = app.getElementsByTagName("service")
    for service in services:
        intents = service.getElementsByTagName("intent-filter")
        for intent in intents:
            actions = intent.getElementsByTagName("action")
            for action in actions:
                return action.getAttribute("android:name")
    return ""

def manifest_find_main_activity(xmlNode, nodeName):
    activities = xmlNode.getElementsByTagName(nodeName)
    for activity in activities:
        intents = activity.getElementsByTagName("intent-filter")
        for intent in intents:
            actions = intent.getElementsByTagName("action")
            for action in actions:
                if("MAIN" in action.getAttribute("android:name")):
                    return activity.getAttribute("android:name")
    return ""

def inform(msg):
    global _args
    print(f"\033[94m[{_args.index} - {_args.last}]\033[0m {msg}")

def args_inquisitor():
    global _args
    parser = argparse.ArgumentParser(description="extract features from apk for RNN GAN, the output is a classic strace file")
    parser.add_argument('apk_path', help='original apk file to process')
    parser.add_argument('-s', '--strace', help="location of strace depot file")
    parser.add_argument('-n', '--vdname', help="virtual device name to use")
    parser.add_argument('-p', '--vdpath', help="genymotion virtual device path")
    parser.add_argument('-t', '--runtime', help="time the apk will be run without doing nothing", type=int)
    parser.add_argument('-m', '--monkey', help="Number of monkey event to trigger")
    parser.add_argument('-i', '--index', help="Index of sample in the db - use to restart a job at a given index - should be set through parent process (stractor client)")
    parser.add_argument('-e', '--last', help="last index of sample assigned to calling instance - for message formatting, mostly -should be set through parent process (stractor client)")
    parser.add_argument('-f', '--nofuzz', help="Don't fuzz intents", action='store_true', default=False)
    _args = parser.parse_args()


if(__name__ == "__main__"):
    if(os.name == "nt"):                                        # Windows compatibility
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        _apktool = "apktool.bat"
    else:
        _apktool = "apktool"

    main()
