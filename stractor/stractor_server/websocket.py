from lib.websocket_server import WebsocketServer
import logging
import json

# OPCODES msg format:
# NUM.Payload
# Client OPCODES
OPCODE_PING = "0"
OPCODE_JOBLIST = "1"
OPCODE_JOBREPPORT = "2"
OPCODE_INFO = "7"
OPCODE_GETJOB = "8"
OPCODE_STARTJOB = "9"


class WebSocket():

    control = None
    slaves = {}

    def __init__(self, addr, port, db):
        self.socket = WebsocketServer(port, host=addr, loglevel=logging.INFO)
        self.socket.set_fn_new_client(self.new_client)
        self.socket.set_fn_message_received(self.msg_rcv)
        self.socket.set_fn_client_left(self.unregister)
        self.db = db

    def new_client(self, client, server):
        self.slaves[client['id']] = {'id': client['id'],
                                     'handler': client['handler'],
                                     'address': client['address'],
                                     'apk': {'idx': 0,
                                             'idx_min': 0,
                                             'idx_max': 0,
                                             'name': ""}
                                     }

        for slave in self.slaves:
            self.socket.send_message(self.slaves[slave], f"{OPCODE_INFO}.NEWCOMER IN TOWN | Currently: {len(self.slaves)}")

    def msg_rcv(self, client, server, message):
        # print(f"Received {message} from {client['address']}")
        # Control HELO
        if(message.startswith("[ctl]")):
            if("helo" in message):
                self.control = client
                del(self.slaves[client['id']])
            if("refr" in message):
                self.refresh_ctl()
            if("whip" in message):
                self.send_all(OPCODE_STARTJOB)
        if(message.startswith(OPCODE_JOBREPPORT)):
            self.update_slave_status(client['id'], message[2:])
        if(message.startswith(OPCODE_GETJOB)):
            self.give_work()
        # print("Current clients:\n{}".format(self.slaves))

    def unregister(self, client, server):
        del(self.slaves[client['id']])
        print("client left")

    def send_all(self, msg):
        self.socket.send_message_to_all(msg)

    def refresh_ctl(self):
        lst = []
        for slave in self.slaves:
            lst.append({'id': slave,
                        'apk': self.slaves[slave]['apk']})
        msg = json.dumps(lst)
        tosend = f"{OPCODE_JOBREPPORT}.{msg}"
        self.socket.send_message(self.control, tosend)

    # right now passing apk_idx only
    def update_slave_status(self, slave_id, status):
        status = json.loads(status)
        self.slaves[slave_id]['apk']['idx'] = status['idx']
        self.slaves[slave_id]['apk']['idx_min'] = status['start']
        self.slaves[slave_id]['apk']['idx_max'] = status['end']
        self.slaves[slave_id]['apk']['name'] = status['name']

    def give_work(self):
        samples = len(self.db)
        sls = len(self.slaves)
        mod = samples % sls

        msgs = [(samples-mod) / sls] * sls

        joblist = []
        tot = 0
        for m in msgs:
            joblist.append((tot, tot + m))
            tot += m + 1

        idx = 0
        for i in range(0, mod):
            msgs[idx % sls] += 1
            idx += 1

        msg_idx = 0
        for slave in self.slaves:
            msg = f'{OPCODE_JOBLIST}.[{joblist[msg_idx][0]},{joblist[msg_idx][1]}]'
            self.socket.send_message(self.slaves[slave], msg)
            msg_idx += 1
