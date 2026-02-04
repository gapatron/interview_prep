"""Tests for 14_computation_exercises solutions."""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "14_computation_exercises"))


def test_conv1d():
    from solution_01_convolution_computation import (
        conv1d_output_length,
        conv1d_num_params,
        conv1d_flops,
    )
    assert conv1d_output_length(100, 5, 2, 1) == (100 + 2 * 1 - 5) // 2 + 1
    assert conv1d_num_params(4, 8, 5, True) == 4 * 8 * 5 + 8
    assert conv1d_flops(4, 8, 5, 50) == 4 * 5 * 8 * 50


def test_conv2d():
    from solution_01_convolution_computation import (
        conv2d_output_shape,
        conv2d_num_params,
        conv2d_flops,
    )
    assert conv2d_output_shape(32, 32, 3, 1, 1) == (32, 32)
    assert conv2d_num_params(3, 64, 3, True) == 3 * 64 * 9 + 64
    assert conv2d_flops(3, 64, 3, 32, 32) == 3 * 9 * 64 * 32 * 32


def test_conv3d():
    from solution_01_convolution_computation import (
        conv3d_output_shape,
        conv3d_num_params,
        conv3d_flops,
    )
    assert conv3d_output_shape(8, 32, 32, (3, 3, 3), 1, 1) == (8, 32, 32)
    assert conv3d_num_params(3, 16, (3, 3, 3), True) == 3 * 16 * 27 + 16
    assert conv3d_flops(3, 16, (3, 3, 3), 8, 32, 32) == 3 * 27 * 16 * 8 * 32 * 32


def test_transformer_shapes_and_params():
    from solution_02_transformer_computation import (
        embedding_shape,
        mha_attention_scores_shape,
        mha_num_params,
        ffn_num_params,
        encoder_block_num_params,
        attention_flops,
    )
    B, L, d_model, H = 2, 128, 768, 12
    assert embedding_shape(B, L, d_model) == (2, 128, 768)
    assert mha_attention_scores_shape(B, H, L) == (2, 12, 128, 128)
    assert mha_num_params(768, 12, True) == 4 * 768 * 768 + 4 * 768
    assert ffn_num_params(768, 3072, True) == 768 * 3072 + 3072 + 3072 * 768 + 768
    assert attention_flops(B, L, d_model) == 2 * B * L * L * d_model


def test_vit2d():
    from solution_03_vision_transformer_computation import (
        vit2d_num_patches,
        vit2d_sequence_length,
        vit2d_patch_embed_output_shape,
        vit2d_patch_embed_num_params,
    )
    assert vit2d_num_patches(224, 224, 16) == 14 * 14
    assert vit2d_sequence_length(224, 224, 16) == 197
    assert vit2d_patch_embed_output_shape(2, 3, 224, 224, 16, 768) == (2, 197, 768)
    assert vit2d_patch_embed_num_params(3, 768, 16, True) == 3 * 768 * 16 * 16 + 768


def test_vit3d():
    from solution_03_vision_transformer_computation import (
        vit3d_num_patches,
        vit3d_sequence_length,
        vit3d_patch_embed_output_shape,
        vit3d_patch_embed_num_params,
    )
    assert vit3d_num_patches(8, 224, 224, (2, 16, 16)) == 4 * 14 * 14
    assert vit3d_sequence_length(8, 224, 224, (2, 16, 16)) == 785
    assert vit3d_patch_embed_output_shape(2, 3, 8, 224, 224, (2, 16, 16), 768) == (2, 785, 768)
    assert vit3d_patch_embed_num_params(3, 768, (2, 16, 16), True) == 3 * 768 * 2 * 16 * 16 + 768


def test_complexity():
    from solution_04_complexity_and_interview_questions import (
        attention_time_complexity,
        attention_space_complexity,
        why_attention_is_expensive,
    )
    assert "L^2" in attention_time_complexity(128, 768)
    assert "L^2" in attention_space_complexity(2, 12, 128)
    assert "L" in why_attention_is_expensive(100)
