import asyncio
import argparse
import websockets
import time
import signal
import sys
import json
import requests
import subprocess
import datetime
from subprocess import PIPE
from enum import Enum


# message opcodes:
# 0: ping
# 1: job list: [start, end]
# 2: job report
# 9: start job
OPCODE_PING = "0"
OPCODE_JOBLIST = "1"
OPCODE_JOBREPPORT = "2"
OPCODE_SLAVEID = "3"
OPCODE_INFO = "7"
OPCODE_GETJOB = "8"
OPCODE_STARTJOB = "9"

JobState = Enum('JobState', 'not_init stopped ready processing process_end job_end')
addr = '192.168.42.211'  # 'localhost'

_failures = []

class Stractor():

    msgs = []

    dbg_jStart = 0
    dbg_jDelta = 0

    def __init__(self, uri, args):
        self.terminate = False
        self.uri = uri
        self.apk_start = 0
        self.apk_end = 0
        self.apk_idx = 0
        self.apk_name = ""
        self.slaveid = -1
        self.state = JobState.not_init
        self.wsess = requests.Session()
        self.proc_handler = None
        self.logfile = None
        self.fail_mode = False
        self.local = args.local
        self.localpath = args.lpath
        self.nofuzz = args.nofuzz

        if(args.force):
            self.slaveid = args.id
            self.apk_start = args.start
            self.apk_end = args.end
            self.apk_idx = args.current
            self.state = JobState.ready
            self.init_log()

    # basically a coroutine
    async def _event_loop(self):
        async with websockets.connect(uri=self.uri, ping_timeout=None) as ws:
            while not self.terminate:
                self.report()
                try:
                    msg_r = await asyncio.wait_for(ws.recv(), timeout=0.25)
                    self.process_msg(msg_r)
                except asyncio.TimeoutError:
                    pass

                if(self.state == JobState.ready):
                    self.start_work()

                if(self.state == JobState.processing):
                    self.process_current_work()

                if(self.state == JobState.process_end):
                    self.get_next_sample()

                if(self.state == JobState.job_end):
                    self.terminate = True

                if(len(self.msgs) > 0):
                    for msg in self.msgs:
                        await ws.send(msg)
                    self.msgs.clear()

                if(self.terminate):
                    print("==== JOB COMPLETE ====\n---- kThxBye!\n\n")
                    await ws.close_connection()
                    return

    def report(self):
        report = json.dumps({'start': self.apk_start,
                             'end': self.apk_end,
                             'idx': self.apk_idx,
                             'name': self.apk_name})
        self.send(report, OPCODE_JOBREPPORT)

    def ask_for_work(self):
        self.send("", OPCODE_GETJOB)

    def start_work(self):
        if(self.local):
            self.get_local_sample(self.apk_idx)
        else:
            self.get_remote_sample(self.apk_idx)
        self.state = JobState.processing

    def process_current_work(self):
        global _failures
        proc_status = -1

        if(self.proc_handler is None):

            # local mode
            if(self.local):
                sample = self.localpath + self.apk_name + "\\" + self.apk_name + ".apk"
            else:
                print("remote fetch to be implemented")

            print("looking for {}".format(sample))

            fuzz = ""
            if(self.nofuzz):
                fuzz = "-f"

            # make sure the worker (here extract Features.py) is somewhere reachable (in your path for instance)
            self.proc_handler = subprocess.Popen(['python', 'extractFeatures.py',
                                                    '-s', "D:\\dataset\\straces",
                                                    '-n', "p"+str(self.slaveid),
                                                    '-p', "C:\\Users\\Hrafn\\AppData\\Local\\Genymobile\\Genymotion\\deployed\\",
                                                    '-t', '15',
                                                    '-m', '1500',
                                                    '-i', str(self.apk_idx),
                                                    '-e', str(self.apk_end),
                                                    fuzz,
                                                    sample])
        else:
            if(proc_status == -1):
                proc_status = self.proc_handler.poll()
            if(proc_status == 0):                   # if proc has ended with success:
                self.proc_handler = None
                self.state = JobState.process_end   # so we can get next sample and start again
            elif(proc_status == 1):
                _failures.append(self.apk_name)
                self.log(f"[FAILURE] - {self.apk_name}")
                self.proc_handler = None
                self.state = JobState.process_end

    def get_next_sample(self):
        self.apk_idx += 1
        if(self.apk_idx > self.apk_end):
            if(len(_failures) == 0):
                self.state = JobState.job_end
                self.apk_idx = -1
                return
            else:
                self.apk_idx = 0
                self.fail_mode = True

        if(self.fail_mode):
            if(self.apk_idx > len(_failures)):
                self.state = JobState.job_end
                self.apk_idx = -1
                return

        if(self.local):
            self.get_local_sample(self.apk_idx)
        else:
            self.get_remote_sample(self.apk_idx)
        # get apk + start process
        self.state = JobState.processing

    def get_local_sample(self, idx):
        if(not self.fail_mode):
            response = requests.post(f"http://{addr}:8000/get_sample", json={"id": str(idx)})
            self.apk_name = response.text
        else:
            self.apk_name = _failures[self.apk_idx]

    def get_remote_sample(self, idx):
        response = requests.post(f"http://{addr}:8000/get_sample", json={"id": str(idx)})
        self.apk_name = response.text
        print(f'Asked for index: {idx} and got: {response.text}')
        apk_path = f'{self.apk_name}/{self.apk_name}.apk'
        apk_bin = requests.get(f'http://{addr}:8000/{apk_path}')
        open(f'tmp/{self.apk_name}.apk', 'wb').write(apk_bin.content)       # need to implement temp directory
        print('apk saved!')

    def init_log(self):
        self.logfile = open(f'p{self.slaveid}.log', 'a')

    def log(self, data):
        now = datetime.datetime.now()
        dt = now.strftime("%Y-%m-%d %H:%M:%S")
        self.logfile.write(f"{dt} : {data}\n")

    def process_msg(self, msg):
        if(msg.startswith(OPCODE_JOBLIST)):     # Get job
            payload = json.loads(msg[2:])
            self.apk_idx = int(payload[0])
            self.apk_start = payload[0]
            self.apk_end = payload[1]
            self.state = JobState.ready
        elif(msg.startswith(OPCODE_STARTJOB)):    # Start job
            if(self.state == JobState.not_init):
                # wait a little to avoid launching everything at the same time
                time.sleep((int(self.slaveid) * 40))
                self.ask_for_work()
            # else: already working...
        elif(msg.startswith(OPCODE_INFO)):
            print(f'[INFO] - {msg[2:]}')
        elif(msg.startswith(OPCODE_SLAVEID)):
            print(f'[INFO] - Assigned id: {msg[2:]}')
            if(self.slaveid == -1):
                self.slaveid = msg[2:]
                self.init_log()
        else:
            print(f"[UNKNOWN] {msg}")

    def start(self):
        asyncio.get_event_loop().run_until_complete(self._event_loop())

    def send(self, data, opcode):
        encoded = f'{opcode}.{str(data)}'
        self.msgs.append(encoded)


#### Class definition end


stractorClient = None


def signal_handler(sig, frame):
    global stractorClient
    # wsClient.send("[FIN]")
    stractorClient.terminate = True
    time.sleep(3)
    sys.exit(0)

def args_inquisitor():
    global _args
    parser = argparse.ArgumentParser(description="Client of stractor server. Will execute a worker on an indexed sample (extractFeatures.py by default, for Android apk straces extraction)")
    parser.add_argument('-s', '--start', help="lower bound of dataset to work with", type=int)
    parser.add_argument('-e', '--end', help="higher bound of dataset to work with", type=int)
    parser.add_argument('-c', '--current', help="start work at this index", type=int)
    parser.add_argument('-i', '--id', help="take this id to work", type=int)
    parser.add_argument('-f', '--force', action='store_true', help="for the client to start the work without waiting for the server to tell it to do so. Helpful when resuming the work of a crashed worker")
    parser.add_argument('-l', '--local', action='store_true', help='dataset is local')
    parser.add_argument('-L', '--lpath', help="Path of local dataset")
    parser.add_argument('-n', '--nofuzz', action="store_true", help="Don't fuzz intents")
    return parser.parse_args()

if(__name__ == "__main__"):
    signal.signal(signal.SIGINT, signal_handler)
    stractorClient = Stractor(f"ws://{addr}:8001", args_inquisitor())
    stractorClient.start()
