"""Jumlah parameter di kartu model: bobot yang dibagi pakai tidak boleh dihitung berkali-kali."""
import torch

from conftest import load_module


def test_unique_param_count_counts_tied_weights_once():
    pr = load_module("package_release_params_under_test", "tools/package_release.py")
    emb = torch.zeros(100, 8)
    sd = {"embedding.weight": emb, "lm_head.weight": emb, "token_embedding.weight": emb, "layer.weight": torch.zeros(8, 8), "layer.bias": torch.zeros(8)}
    assert sum(t.numel() for t in sd.values()) == 3 * 800 + 64 + 8        # penjumlahan lama: embedding terhitung 3 kali
    assert pr.unique_param_count(sd) == 800 + 64 + 8


def test_unique_param_count_matches_the_real_model():
    pr = load_module("package_release_params_real_under_test", "tools/package_release.py")
    emb = torch.nn.Embedding(50, 4)
    head = torch.nn.Linear(4, 50, bias=False)
    head.weight = emb.weight                                              # weight tying
    model = torch.nn.ModuleDict({"emb": emb, "head": head})
    assert pr.unique_param_count(model.state_dict()) == sum(p.numel() for p in model.parameters()) == 200
