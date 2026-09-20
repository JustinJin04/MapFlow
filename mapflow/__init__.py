from mapflow.client import ClientReceiver, ClientSender
from mapflow.core import load_weights
from mapflow.core.prof_marker import prof_marker


def get_mapflow_client(
    weights_dir,
    num_layers,
    num_ignored_layers,
    num_send_layers,
    num_total_heads, # before tp partition
    num_send_heads, # before tp partition
    tp_rank,
    tp_size,
    head_dim,
    max_seq_len,
    block_size,
    top_p,
    server_main_port,
    run_warmup=True,
    force_wait_sync=False,
):
    """
    for sender, weights_dir is None
    for receiver, weights_dir has model_layers_{layer_idx}_self_attn_attn_weight.pt
    """
    
    if weights_dir is None:
        # sender
        client = ClientSender(
            num_layers=num_layers,
            num_ignored_layers=num_ignored_layers,
            num_total_heads=num_total_heads,
            tp_rank=tp_rank,
            tp_size=tp_size,
            head_dim=head_dim,
            block_size=block_size,
            top_p=top_p,
            server_main_port=server_main_port,
            device=f"cuda:{tp_rank}"
        )
    else:
        # receiver
        # if run_warmup:
        #     run_warmup_once(
        #         m=num_send_heads,
        #         n=num_total_heads,
        #     )
        weights_dict = load_weights(weights_dir)
        if force_wait_sync is None:
            force_wait_sync = False
        client = ClientReceiver(
            num_layers=num_layers,
            weights=weights_dict,
            num_heads=num_total_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_send_layers=num_send_layers,
            server_main_port=server_main_port,
            max_seq_len=max_seq_len,
            num_ignored_layers=num_ignored_layers,
            force_wait_sync=force_wait_sync,
        )
    return client
