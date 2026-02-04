"""
14 — Complexity and interview questions
=========================================
Learning goal: Time/space complexity of attention vs convolution, FLOPs, memory;
and answers to tricky interview questions (so you can rehearse and master them).

Implement the functions that return complexity answers (so you practice writing them);
then read the interview_questions dict and rehearse answering aloud.
"""

from typing import Tuple


# ---------------------------------------------------------------------------
# Complexity (Big-O style: what dominates?)
# ---------------------------------------------------------------------------


def attention_time_complexity(L: int, d_model: int) -> str:
    """
    Time complexity of self-attention (one layer) in terms of L and d_model.
    Q@K^T is (B,L,d) @ (B,d,L) → O(B*L^2*d). attn@V is O(B*L^2*d).
    So overall O(L^2 * d) per batch item. Return "O(L^2 * d)" or "quadratic in L".
    """
    # TODO: return the dominant term, e.g. "O(L^2 * d_model)"
    raise NotImplementedError


def attention_space_complexity(B: int, H: int, L: int) -> str:
    """
    Space (memory) for storing attention scores (before softmax).
    Shape (B, H, L, L). So O(B * H * L^2). Return "O(B * H * L^2)".
    """
    # TODO: return the dominant term
    raise NotImplementedError


def conv2d_time_complexity_per_layer(
    C_in: int, C_out: int, K: int, H_out: int, W_out: int
) -> str:
    """
    FLOPs for one Conv2d layer: C_in * K^2 * C_out * H_out * W_out.
    In Big-O: O(C_in * C_out * K^2 * H * W). Return "O(C_in * C_out * K^2 * H * W)".
    """
    # TODO: return the dominant term (use "K" for kernel size)
    raise NotImplementedError


def why_attention_is_expensive(L: int) -> str:
    """
    One-sentence reason: attention does what with L?
    Answer: pairwise comparisons over L positions → L^2.
    """
    # TODO: return a short string explaining why attention is O(L^2)
    raise NotImplementedError


def conv_vs_attention_compute(
    L_conv: int,
    C: int,
    K: int,
    L_seq: int,
    d: int,
) -> Tuple[str, str]:
    """
    Compare one conv1d layer (input length L_conv, C channels, kernel K) vs one attention layer (seq L_seq, dim d).
    Conv FLOPs: C*C*K*L_conv (assuming C_in=C_out=C). Attention FLOPs: ~2*L_seq^2*d.
    Return ("conv_flops_formula", "attention_flops_formula") as strings for interview explanation.
    """
    # TODO: return ("C*C*K*L_conv", "2*L_seq^2*d") or similar
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Interview Q&A (read and rehearse; no code required)
# ---------------------------------------------------------------------------

INTERVIEW_QUESTIONS = {
    "Q1: Output shape of Conv2d": {
        "question": "Given input (B, C_in, H, W), kernel size K, stride S, padding P, what is (H_out, W_out)?",
        "answer": "H_out = floor((H + 2*P - K) / S) + 1, same for W_out. Example: 32x32, K=3, S=1, P=1 → 32x32.",
    },
    "Q2: Parameters in Conv2d": {
        "question": "How many parameters in Conv2d(C_in, C_out, kernel_size=3)?",
        "answer": "C_in * C_out * 3 * 3 + C_out (bias). No dependence on H, W.",
    },
    "Q3: Why is self-attention O(L^2)?": {
        "question": "Why does self-attention scale quadratically with sequence length?",
        "answer": "We compute attention scores between every pair of positions: L×L matrix. So time and space scale as L^2.",
    },
    "Q4: Multi-head attention parameter count": {
        "question": "How many parameters in multi-head attention (d_model, num_heads)? No extra params for 'heads'—why?",
        "answer": "4 * d_model^2 + 4*d_model (W_q, W_k, W_v, W_out, each d_model×d_model). Heads are a reshape of the same weights; head_dim = d_model/num_heads.",
    },
    "Q5: ViT sequence length from image": {
        "question": "Image 224×224, patch size 16. What is the sequence length (including [CLS])?",
        "answer": "Num patches = (224/16)^2 = 14*14 = 196. With [CLS], L = 197.",
    },
    "Q6: ViT for video": {
        "question": "Video (T=16, H=224, W=224), patch (2,16,16). How many patches?",
        "answer": "(16/2)*(224/16)*(224/16) = 8*14*14 = 1568. L = 1569 with [CLS].",
    },
    "Q7: Conv3d output shape": {
        "question": "Input (B, C, T, H, W) = (2, 3, 8, 32, 32), Conv3d(3, 64, kernel_size=3, padding=1). Output shape?",
        "answer": "(2, 64, 8, 32, 32)—same spatial/temporal size when padding=1, stride=1.",
    },
    "Q8: FLOPs of one attention layer": {
        "question": "Rough FLOPs for one self-attention layer: batch B, sequence L, dimension d?",
        "answer": "Q@K^T: B*L*d*L = B*L^2*d. attn@V: B*L*L*d = B*L^2*d. Total ~2*B*L^2*d (mult-adds).",
    },
    "Q9: LayerNorm in transformer": {
        "question": "How many parameters in LayerNorm(d_model)?",
        "answer": "2*d_model: one weight (scale) and one bias (shift), both of length d_model.",
    },
    "Q10: Patch embedding parameters": {
        "question": "ViT patch embedding: Conv2d(3, 768, kernel_size=16, stride=16). How many parameters?",
        "answer": "3 * 768 * 16 * 16 + 768 = 589,824 + 768 = 590,592.",
    },
}


def get_question(key: str) -> str:
    """Return the question text for a given key."""
    return INTERVIEW_QUESTIONS[key]["question"]


def get_answer(key: str) -> str:
    """Return the answer text for a given key."""
    return INTERVIEW_QUESTIONS[key]["answer"]


if __name__ == "__main__":
    assert "L^2" in attention_time_complexity(128, 768)
    assert "L^2" in attention_space_complexity(2, 12, 128)
    assert "K" in conv2d_time_complexity_per_layer(64, 64, 3, 32, 32)
    assert "pairwise" in why_attention_is_expensive(100).lower() or "L^2" in why_attention_is_expensive(100)
    formulas = conv_vs_attention_compute(100, 64, 3, 100, 768)
    assert len(formulas) == 2

    print("04_complexity_and_interview_questions OK.")
    print("Rehearse INTERVIEW_QUESTIONS keys:", list(INTERVIEW_QUESTIONS.keys()))
