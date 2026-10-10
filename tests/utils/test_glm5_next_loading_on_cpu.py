# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2026 wht2139
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metadata-only target preflight and random tiny weights; never fetch target weights."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

MODEL_ID = "zai-org/GLM-5.3-Flash-BF16"
MODEL_REVISION = "a5b45eb41df6402735dedc900be14a42e8d5e538"


def tiny_config():
    # Keep KDA, full DSA, mHC and routed/shared experts, including the vision shell.
    return AutoConfig.for_model(
        "glm5_next",
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_shared_experts=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            kv_lora_rank=8,
            q_lora_rank=8,
            qk_rope_head_dim=0,
            qk_nope_head_dim=16,
            v_head_dim=16,
            n_group=1,
            topk_group=1,
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=2,
            layer_types=["linear_attention", "indexed_attention"],
            indexer_types=["full", "full"],
            mlp_layer_types=["sparse", "sparse"],
            index_topk=4,
            index_kpool=2,
            index_head_dim=16,
            index_n_heads=2,
            linear_head_dim=16,
            linear_num_heads=2,
            linear_conv_kernel_dim=4,
            hc_mult=4,
            hc_sinkhorn_iters=4,
            use_cache=False,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            num_heads=4,
            out_hidden_size=32,
            intermediate_size=64,
            projection_intermediate_size=64,
            image_size=16,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        image_token_id=60,
        video_token_id=61,
        image_start_token_id=56,
        image_end_token_id=57,
        video_start_token_id=58,
        video_end_token_id=59,
    )


def test_target_metadata_selects_glm5_next():
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    assert config.model_type == "glm5_next"
    assert type(config).__name__ == "Glm5NextConfig"
    # Retain the snapshot's config class, but never instantiate its 320B dimensions.
    small = type(config).from_dict(tiny_config().to_dict())
    model = AutoModelForImageTextToText.from_config(small, attn_implementation="eager")
    assert type(model).__name__ == "Glm5NextForConditionalGeneration"
    assert model.can_generate()


def test_tiny_text_logits_save_reload(tmp_path):
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(tiny_config(), attn_implementation="eager").cpu().eval()
    tokens = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
    assert logits.shape == (1, 4, 64)
    assert torch.isfinite(logits).all()
    assert any(parameter.requires_grad for parameter in model.parameters())
    model.save_pretrained(tmp_path)
    restored = (
        AutoModelForImageTextToText.from_pretrained(tmp_path, local_files_only=True, attn_implementation="eager")
        .cpu()
        .eval()
    )
    assert type(restored).__name__ == "Glm5NextForConditionalGeneration"
    with torch.no_grad():
        actual = restored(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
    torch.testing.assert_close(actual, logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_qwen_cpu_save_reload(tmp_path, model_type):
    config = AutoConfig.for_model(
        model_type,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").cpu().eval()
    tokens = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        expected = model(input_ids=tokens).logits
    model.save_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True).cpu().eval()
    with torch.no_grad():
        actual = restored(input_ids=tokens).logits
    torch.testing.assert_close(actual, expected)
