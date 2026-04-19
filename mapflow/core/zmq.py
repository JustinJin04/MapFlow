import zmq
import pickle
from typing import Optional

class ZMQCommunicator:
    def __init__(self, my_port: int|None = None):
        self.context = zmq.Context()
        
        self.receiver = self.context.socket(zmq.PULL)
        
        if my_port is None:
            self.my_port = self.receiver.bind_to_random_port("tcp://127.0.0.1")
        else:
            self.my_port = my_port
            self.receiver.bind(f"tcp://127.0.0.1:{self.my_port}")
            
        self.senders = {}

    def send(self, content: dict, dst_port: int):
        if dst_port not in self.senders:
            sender = self.context.socket(zmq.PUSH)
            sender.connect(f"tcp://127.0.0.1:{dst_port}")
            self.senders[dst_port] = sender
        
        content["src_port"] = self.my_port  # 此时 my_port 必定有确切的数值
        self.senders[dst_port].send(pickle.dumps(content))

    def recv(self):
        msg = self.receiver.recv()
        return pickle.loads(msg)
