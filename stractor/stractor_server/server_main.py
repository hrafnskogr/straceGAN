#! /usr/bin/env python3
# Samples
# 0bc37d923c7b237346f7a4984aad8847
# 310d73958043bca1bcc27eb70997db8e
# 5b07d87a869df3ab773488fad99015f2
# 68f7b1e5352331464462aceaf1ccfda3
# b341b299f572c04319294152b2682857

import time
from http.server import HTTPServer
from webserver import *
from websocket import WebSocket
import multiprocessing


_host_name = '192.168.1.211'  # 'localhost'
_port_number = 8000


def db_init(csv):
    tmp = {}
    ite = 0
    with open(csv, 'r') as dbfile:
        next(dbfile)
        for line in dbfile:
            name, h = line.split(";")
            tmp[ite] = {'name': name.rstrip(), 'hash':  h.strip()}
            ite += 1
    return tmp


if(__name__ == '__main__'):
    db = db_init("db/sampledb.csv")

    httpd = WebServer(db, (_host_name, _port_number), WebServerRequestHandler)
    wsd = WebSocket(_host_name, _port_number+1, db)

    hs = multiprocessing.Process(target=httpd.serve_forever)
    ws = multiprocessing.Process(target=wsd.socket.serve_forever)

    shutdown = False

    try:
        print(time.asctime(), f'Server UP - {_host_name}:{_port_number}')
        hs.start()
        ws.start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass

    hs.terminate()
    ws.terminate()

    hs.join()
    ws.join()

    httpd.server_close()
    wsd.socket.server_close()
    print(time.asctime(), f'Server DOWN - {_host_name}:{_port_number}')
