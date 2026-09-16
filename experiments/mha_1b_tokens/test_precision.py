"""CPU BF16 autocast references; does not substitute for compiled CUDA tests."""
import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from .model import ModernMHADecoder, GrainMLP, copy_to_hf


def tiny_config():
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())['model']
    cfg.update(hidden_size=32, intermediate_size=48, head_dim=2,
               num_attention_heads=16, num_key_value_heads=16,
               num_hidden_layers=2, vocab_size=128)
    return cfg


def independent_grain(x, numerator, denominator, groups, eps):
    """Direct powers definition, independent of fallback/Horner/fused code."""
    shape = x.shape
    grouped = x.reshape(*shape[:-1], groups, shape[-1] // groups)
    rms = (grouped.square().mean(-1, keepdim=True) + eps).sqrt()
    unit = grouped / rms
    powers = torch.stack([unit ** degree for degree in range(6)], -1)
    num = (powers * numerator.reshape(*([1] * (unit.ndim-2)), groups, 1, 6)).sum(-1)
    positive = denominator.abs()
    den = torch.ones_like(unit)
    for degree in range(1, 5):
        den = den + positive[:, degree-1].reshape(*([1]*(unit.ndim-2)), groups, 1) * unit.abs() ** degree
    return (rms * num / den).reshape(shape)


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1337)

    def assert_fp32_finite_gradients(self, module):
        for name, parameter in module.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32, name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertEqual(parameter.grad.dtype, torch.float32, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

    def test_grain_autocast_boundary_and_independent_algebra(self):
        with patch.dict(os.environ, {'RATIONAL_OPT_TORCH_FALLBACK': '1'}):
            mlp = GrainMLP(tiny_config(), groups=3)
            reference = copy.deepcopy(mlp)
            x = torch.randn(2, 16, 32, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            observed = {}
            def capture(module, args, output):
                observed.update(input=args[0].detach().clone(), output=output.detach().clone(),
                                autocast=torch.is_autocast_enabled('cpu'))
            handle = mlp.rlb_activation.register_forward_hook(capture)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = mlp(x)
                projected = reference.in_proj(xr)
                with torch.autocast('cpu', enabled=False):
                    a = reference.rlb_activation
                    features = independent_grain(projected.float(), a.numerator, a.denominator, a.groups, a.eps)
                expected = reference.out_proj(features)
            handle.remove()
            self.assertEqual(projected.dtype, torch.bfloat16)
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertEqual(observed['input'].dtype, torch.float32)
            self.assertEqual(observed['output'].dtype, torch.float32)
            self.assertFalse(observed['autocast'])
            torch.testing.assert_close(observed['input'], projected.float(), rtol=0, atol=0)
            torch.testing.assert_close(observed['output'], features, rtol=2e-6, atol=3e-7)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            cotangent = torch.randn_like(actual)
            (actual.float()*cotangent.float()).sum().backward()
            (expected.float()*cotangent.float()).sum().backward()
            self.assert_fp32_finite_gradients(mlp)
            for (name,p),(_,q) in zip(mlp.named_parameters(),reference.named_parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5, msg=name)
            torch.testing.assert_close(x.grad, xr.grad, atol=2e-5, rtol=2e-5)

    def test_mha_autocast_hf_reference(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = tiny_config()
        model = ModernMHADecoder(cfg)
        reference = Qwen3ForCausalLM(Qwen3Config(**cfg))
        reference.config._attn_implementation = 'sdpa'
        copy_to_hf(model, reference)
        ids = torch.randint(0, 128, (2, 16))
        targets = ids.roll(-1, 1)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = model(ids)
            expected = reference(ids, use_cache=False).logits
            actual_loss = F.cross_entropy(actual.float().reshape(-1,128), targets.reshape(-1))
            expected_loss = F.cross_entropy(expected.float().reshape(-1,128), targets.reshape(-1))
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-2)
        torch.testing.assert_close(actual_loss, expected_loss, atol=2e-4, rtol=2e-4)
        actual_loss.backward(); expected_loss.backward()
        self.assert_fp32_finite_gradients(model)
        self.assert_fp32_finite_gradients(reference)
        torch.testing.assert_close(model.embed_tokens.weight.grad, reference.model.embed_tokens.weight.grad,
                                   atol=5e-4, rtol=2e-2)


if __name__ == '__main__':
    unittest.main()
