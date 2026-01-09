from kvsched.models.request import InferenceRequest, RequestProfile, RequestRuntime, KVCacheState, Stage
from kvsched.models.node import NodeState, GPUDevice
from kvsched.models.network import NetworkModel, NetworkTopology, NetworkLink

def test_request_builds():
    r = InferenceRequest(
        request_id="r1",
        profile=RequestProfile(model_id="m", prompt_tokens=0, max_new_tokens=1),
        runtime=RequestRuntime(stage=Stage.PREFILL, seq_len=0, generated_tokens=0),
        kv=KVCacheState(owner_node_id="n1", kv_bytes=0, num_cached_tokens=0),
    )
    assert r.request_id == "r1"

def test_node_builds():
    n = NodeState(node_id="n1", gpu=GPUDevice(name="gpu", vram_total_bytes=1, vram_free_bytes=1))
    assert n.can_host_kv(1)

def test_network_builds():
    topo = NetworkTopology()
    topo.add_link(NetworkLink(src="a", dst="b", bandwidth_Gbps=10.0, rtt_ms=1.0))
    net = NetworkModel(topo=topo)
    assert net.est_transfer_ms("a","b",1024) > 0
