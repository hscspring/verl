from unittest.mock import Mock

from verl.workers.engine.veomni import transformer_impl


def test_load_checkpoint_releases_cuda_cache_after_offload(monkeypatch):
    events = []
    engine = transformer_impl.VeOmniEngine.__new__(transformer_impl.VeOmniEngine)
    engine._is_offload_param = True
    engine._is_offload_optimizer = True
    engine._uses_fsdp2_cpu_offload_policy = False
    engine.module = object()
    engine.optimizer = object()
    engine.checkpoint_manager = Mock()
    engine.checkpoint_manager.load_checkpoint.side_effect = lambda **_: events.append("restore")

    monkeypatch.setattr(
        transformer_impl,
        "load_veomni_model_to_gpu",
        lambda _: events.append("model_to_gpu"),
    )
    monkeypatch.setattr(
        transformer_impl,
        "offload_veomni_model_to_cpu",
        lambda _: events.append("model_to_cpu"),
    )
    monkeypatch.setattr(
        transformer_impl,
        "offload_veomni_optimizer",
        lambda _: events.append("optimizer_to_cpu"),
    )
    monkeypatch.setattr(
        transformer_impl,
        "aggressive_empty_cache",
        lambda *, force_sync: events.append(("empty_cache", force_sync)),
    )
    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *_, **__: None)
    monkeypatch.setattr(
        transformer_impl.torch.distributed,
        "barrier",
        lambda: events.append("barrier"),
    )

    engine.load_checkpoint("/checkpoint", del_local_after_load=False)

    assert events == [
        "model_to_gpu",
        "restore",
        "barrier",
        "model_to_cpu",
        "optimizer_to_cpu",
        ("empty_cache", True),
    ]
