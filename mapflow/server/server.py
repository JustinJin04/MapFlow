import argparse
import traceback
import torch
import torch.multiprocessing as mp
from mapflow.server.sessions import SendSessionWorker, RecvSessionWorker
from mapflow.server.block_manager import BlockManager
from mapflow.core.zmq import ZMQCommunicator
from mapflow.core.ipc_utils import CudaIPCWrapper

mp_ctx = mp.get_context('spawn')

class CacheServer:
    def __init__(
        self,
        num_sender_layers: int,
        num_ignored_layers: int,
        num_total_heads: int,  # before tp partition
        tp_size: int,
        block_size: int,
        max_num_reqs: int,
        dtype: str,
        port: int,
        max_num_blocks: int,
    ):
        if dtype == "float16":
            self.torch_dtype = torch.float16
        elif dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            assert 0
        self.send_layer_list = list(range(num_ignored_layers, num_sender_layers-1))
        self.tp_size = tp_size
        self.max_num_reqs = max_num_reqs
        self.block_size = block_size
        self.num_total_heads = num_total_heads
        self.num_heads = num_total_heads // tp_size

        # Initialize Shared block_manager
        self.block_manager = BlockManager(
            layer_list=self.send_layer_list,
            block_size=block_size,
            dtype=self.torch_dtype,
            tp_size=tp_size,
            max_num_blocks=max_num_blocks,
        )

        # ZMQ endpoint
        self.comm = ZMQCommunicator(port)

        # wait for all tp workers to register
        self.pending_senders = []

        self.recv_session_list = []
        

    def run(self):
        while True:
            try:
                msg = self.comm.recv()
                cmd = msg["cmd"]
                if cmd == "REGISTER_SENDER":
                    self.pending_senders.append({
                        "rank": msg["rank"],
                        "port": msg["src_port"],
                        "crow_ipc": msg["crow_ipc"],
                        "col_ipc": msg["col_ipc"],
                        "block_ipc": msg["block_ipc"],
                    })

                    if len(self.pending_senders) == self.tp_size:
                        port_queue = mp_ctx.Queue()
                        self.sender_session = SendSessionWorker(
                            send_layer_list=self.send_layer_list,
                            queue=self.block_manager.store_queue,
                            data_blocks=self.block_manager.data_blocks,
                            tp_size=self.tp_size,
                            num_heads=self.num_heads,
                            sender_info_list=self.pending_senders,
                            port_queue=port_queue
                        )
                        self.sender_session.start()
                        session_port = port_queue.get()
                        for s in self.pending_senders:
                            self.comm.send({"session_port": session_port}, s["port"])
                        self.pending_senders = []

                elif cmd == "REGISTER_RECEIVER":
                    ipc_wrapper = msg["ipc_wrapper"]
                    port_queue = mp_ctx.Queue()
                    receiver_id = len(self.recv_session_list)
                    print(f"recv register. layer_list: {list(ipc_wrapper['layers'].keys())}", flush=True)
                    recv_session = RecvSessionWorker(
                        send_layer_list=list(ipc_wrapper["layers"].keys()),
                        ipc_wrapper=ipc_wrapper,
                        max_num_reqs=self.max_num_reqs,
                        queue=self.block_manager.retrieve_queue_list[receiver_id],
                        port_queue=port_queue
                    )
                    recv_session.start()
                    self.recv_session_list.append(recv_session)
                    server_thread_port = port_queue.get()
                    self.comm.send({
                        "server_thread_port": server_thread_port,
                        "data_blocks_ipc": CudaIPCWrapper(self.block_manager.data_blocks),
                    }, msg["src_port"])

                else:
                    raise ValueError(f"Unknown cmd: {cmd}")
            except Exception as e:
                print(f"[CacheServer] Exception occurred: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    torch.cuda.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sender_layers", type=int, required=True)
    parser.add_argument("--num_ignored_layers", type=int, required=True)
    parser.add_argument("--num_sender_heads", type=int, required=True)
    parser.add_argument("--tp_size", type=int, required=True)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--max_num_reqs", type=int, default=64)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], required=True)
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--max_num_blocks", type=int, default=4096000)
    args = parser.parse_args()
    
    server = CacheServer(
        num_sender_layers=args.num_sender_layers,
        num_ignored_layers=args.num_ignored_layers,
        num_total_heads=args.num_sender_heads,
        tp_size=args.tp_size,
        block_size=args.block_size,
        max_num_reqs=args.max_num_reqs,
        dtype=args.dtype,
        port=args.port,
        max_num_blocks=args.max_num_blocks,
    )
    server.run()
