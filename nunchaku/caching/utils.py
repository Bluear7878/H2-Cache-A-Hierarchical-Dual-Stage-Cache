# This caching functionality is largely brought from https://github.com/chengzeyi/ParaAttention/src/para_attn/first_block_cache/

import contextlib
import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, Optional, Tuple

import torch
from torch import nn

from nunchaku.models.transformers.utils import pad_tensor

from nunchaku.caching.FBthresholding import *
import torch.nn.functional as F

MULTI_POOL_SIZES = 2048
SINGLE_POOL_SIZES = 384
pool_fn=F.avg_pool2d
#pool_fn=F.max_pool2d

num_transformer_blocks = 19  # FIXME
num_single_transformer_blocks = 38  # FIXME

import matplotlib.pyplot as plt
import os

# --- 1. 각 분야에 대한 전역 변수 선언 ---
MULT_DIFF_HISTORY = []
MULT_THRESHOLD_HISTORY = []
MULT_CACHE_HIT = 0
SING_DIFF_HISTORY = []
SING_THRESHOLD_HISTORY = []
SING_CACHE_HIT = 0


# --- 2. 모든 전역 이력을 초기화하는 함수 ---
def clear_cache_history():
    """다음 실행을 위해 모든 전역 이력을 초기화합니다."""
    global MULT_DIFF_HISTORY, MULT_THRESHOLD_HISTORY, SING_DIFF_HISTORY, SING_THRESHOLD_HISTORY
    MULT_DIFF_HISTORY.clear()
    MULT_THRESHOLD_HISTORY.clear()
    SING_DIFF_HISTORY.clear()
    SING_THRESHOLD_HISTORY.clear()
    print("Global cache histories cleared.")
    
from matplotlib.lines import Line2D
def save_cache_history_plots(base_save_path="outputs/global_plot"):
    # 공통 스타일
    label_font = {'fontsize': 16, 'fontweight': 'bold'}
    tick_params = {'axis': 'both', 'which': 'major', 'labelsize': 14, 'width': 1.5}
    legend_kwargs = {
        'prop': {'weight': 'bold', 'size': 14},
        'frameon': True
    }

    def plot_with_hits(ax, diffs, threshs):
        steps = list(range(len(diffs)))

        # 값 선으로 연결
        ax.plot(steps, diffs,
                linestyle='-',
                linewidth=2,
                color='blue',
                label='Value')

        # 히트/미스 마스크
        hits = [d >= t for d, t in zip(diffs, threshs)]
        # 히트 포인트 (큰 원, 테두리 두께)
        ax.scatter(
            [s for s, h in zip(steps, hits) if h],
            [d for d, h in zip(diffs, hits) if h],
            marker='o',
            s=150,              # 크게 조정
            linewidths=2,       # 테두리 두께
            edgecolors='black',
            color='green',
            label='Cache Hit'
        )
        # 미스 포인트 (큰 엑스, 테두리 두께)
        ax.scatter(
            [s for s, h in zip(steps, hits) if not h],
            [d for d, h in zip(diffs, hits) if not h],
            marker='x',
            s=150,              # 크게 조정
            linewidths=3,       # 선 두께
            color='red',
            label='Cache Miss'
        )

        # Threshold 라인
        ax.plot(steps, threshs,
                linestyle='--',
                linewidth=2,
                color='black',
                label='Threshold')

        # 축 레이블
        ax.set_xlabel("Inference Step", **label_font)
        ax.set_ylabel("Value", **label_font)
        ax.tick_params(**tick_params)

        # 범례: 순서 지정, 볼드체 적용, 마커 사이즈 조정
        handles = [
            Line2D([], [], color='blue', linestyle='-', linewidth=2, label='Value'),
            Line2D([], [], marker='o', color='green', markeredgecolor='black',
                   linestyle='None', markersize=12, markeredgewidth=2, label='Cache Miss'),
            Line2D([], [], marker='x', color='red',
                   linestyle='None', markersize=12, markeredgewidth=3, label='Cache Hit'),
            Line2D([], [], color='black', linestyle='--', linewidth=2, label='Threshold'),
        ]
        ax.legend(handles=handles, **legend_kwargs)
        ax.grid(True, linewidth=0.5, linestyle='--')

    # Multi-block
    if MULT_DIFF_HISTORY:
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_with_hits(ax, MULT_DIFF_HISTORY, MULT_THRESHOLD_HISTORY)
        plt.tight_layout()
        path = f"{base_save_path}_multi.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.close(fig)
        print(f"✅ Multi-block plot saved to: {path}")
    else:
        print("No 'Multi-block' history to plot.")

    # Single-block
    if SING_DIFF_HISTORY:
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_with_hits(ax, SING_DIFF_HISTORY, SING_THRESHOLD_HISTORY)
        plt.tight_layout()
        path = f"{base_save_path}_single.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300)
        plt.close(fig)
        print(f"✅ Single-block plot saved to: {path}")
    else:
        print("No 'Single-block' history to plot.")

@dataclasses.dataclass
class CacheContext:
    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    def get_incremental_name(self, name=None):
        if name is None:
            name = "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    def reset_incremental_name(self):
        self.incremental_name_counters.clear()

    # @torch.compiler.disable # This is a torchscript feature
    def get_buffer(self, name=str):
        return self.buffers.get(name)

    def set_buffer(self, name, buffer):
        self.buffers[name] = buffer

    def clear_buffers(self):
        self.buffers.clear()


@torch.compiler.disable
def get_buffer(name):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_buffer(name)


@torch.compiler.disable
def set_buffer(name, buffer):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.set_buffer(name, buffer)


_current_cache_context = None


def create_cache_context():
    return CacheContext()


def get_current_cache_context():
    return _current_cache_context


@contextlib.contextmanager
def cache_context(cache_context):
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context
    try:
        yield
    finally:
        _current_cache_context = old_cache_context

@torch.compiler.disable
def are_two_tensors_similar(
    t1: torch.Tensor,
    t2: torch.Tensor,
    *,
    threshold: float,
    parallelized=False,
    mode: str = "multi"
):
    
    if mode == "multi":
        pool_size = MULTI_POOL_SIZES
    else:
        pool_size = SINGLE_POOL_SIZES
    if t2.ndim != 4:
        t1 = t1.view(t1.shape[0], 1, t1.shape[1], -1)
        t2 = t2.view(t2.shape[0], 1, t2.shape[1], -1)

    B, C, H, W = t2.shape
    
    stride = H // pool_size
    thumbnail1 = pool_fn(t1, kernel_size=stride, stride=stride)
    thumbnail2 = pool_fn(t2, kernel_size=stride, stride=stride)

    mean_diff = (thumbnail1 - thumbnail2).abs().mean()
    mean_t1   = thumbnail1.abs().mean()
    diff      = (mean_diff / (mean_t1)).item()    

    return diff < threshold, diff


@torch.compiler.disable
def apply_prev_hidden_states_residual(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    mode: str = "multi",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "multi":
        hidden_states_residual = get_buffer("multi_hidden_states_residual")
        assert hidden_states_residual is not None, "multi_hidden_states_residual must be set before"
        hidden_states = hidden_states + hidden_states_residual
        hidden_states = hidden_states.contiguous()

        if encoder_hidden_states is not None:
            enc_hidden_res = get_buffer("multi_encoder_hidden_states_residual")
            msg = "multi_encoder_hidden_states_residual must be set before"
            assert enc_hidden_res is not None, msg
            encoder_hidden_states = encoder_hidden_states + enc_hidden_res
            encoder_hidden_states = encoder_hidden_states.contiguous()

        return hidden_states, encoder_hidden_states

    elif mode == "single":
        single_residual = get_buffer("single_hidden_states_residual")
        msg = "single_hidden_states_residual must be set before"
        assert single_residual is not None, msg
        hidden_states = hidden_states + single_residual
        hidden_states = hidden_states.contiguous()

        return hidden_states

    else:
        raise ValueError(f"Unknown mode {mode}; expected 'multi' or 'single'")


@torch.compiler.disable
def get_can_use_cache(
    first_hidden_states_residual: torch.Tensor, threshold: float, parallelized: bool = False, mode: str = "multi"
):
    if mode == "multi":
        buffer_name = "first_multi_hidden_states_residual"
    elif mode == "single":
        buffer_name = "first_single_hidden_states_residual"
    else:
        raise ValueError(f"Unknown mode {mode}; expected 'multi' or 'single'")

    prev_res = get_buffer(buffer_name)

    if prev_res is None:
        return False, threshold

    is_similar, diff = are_two_tensors_similar(
        prev_res,
        first_hidden_states_residual,
        threshold=threshold,
        parallelized=parallelized,
        mode=mode,
    )
    return is_similar, diff


def check_and_apply_cache(
    *,
    first_residual: torch.Tensor,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    max_samples: int = 4096*2,
    threshold: float,
    parallelized: bool,
    mode: str,
    verbose: bool,
    call_remaining_fn,
    Force_hit : False,
    remaining_kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
    if Force_hit is True:
        can_use_cache = True
        diff = threshold
    else:
        can_use_cache, diff = get_can_use_cache(
            first_residual,
            threshold=threshold,
            parallelized=parallelized,
            mode=mode,
        )
    global MULT_CACHE_HIT, SING_CACHE_HIT
    if verbose:
        if mode == "multi":
            MULT_DIFF_HISTORY.append(diff)
            MULT_THRESHOLD_HISTORY.append(threshold)
        elif mode == "single":
            SING_DIFF_HISTORY.append(diff)
            SING_THRESHOLD_HISTORY.append(threshold)
    
    torch._dynamo.graph_break()

    if can_use_cache:
        if verbose:
            if mode == "multi":
                MULT_CACHE_HIT+=1
            else:
                SING_CACHE_HIT+=1
            print(f"[{mode.upper()}] Cache hit! diff={diff:.4f}, " f"new threshold={threshold:.4f}")

        out = apply_prev_hidden_states_residual(hidden_states, encoder_hidden_states, mode=mode)
        updated_h, updated_enc = out if isinstance(out, tuple) else (out, None)
        return updated_h, updated_enc, threshold, can_use_cache
    
    old_threshold = threshold
    threshold = diff #adatptive

    if verbose:
        print(f"[{mode.upper()}] Cache miss. diff={diff:.4f}, " f"was={old_threshold:.4f} => now={threshold:.4f}")
    
    if mode == "multi":
        set_buffer("first_multi_hidden_states_residual", first_residual)
    else:
        set_buffer("first_single_hidden_states_residual", first_residual)

    result = call_remaining_fn(
        hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, **remaining_kwargs
    )

    if mode == "multi":
        updated_h, updated_enc, hs_res, enc_res = result
        set_buffer("multi_hidden_states_residual", hs_res)
        set_buffer("multi_encoder_hidden_states_residual", enc_res)
        return updated_h, updated_enc, threshold, can_use_cache

    elif mode == "single":
        updated_cat_states, cat_res = result
        set_buffer("single_hidden_states_residual", cat_res)
        return updated_cat_states, None, threshold, can_use_cache

    raise ValueError(f"Unknown mode {mode}")


class SanaCachedTransformerBlocks(nn.Module):#
    def __init__(
        self,
        *,
        transformer=None,
        residual_diff_threshold,
        verbose: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer.transformer_blocks
        self.residual_diff_threshold = residual_diff_threshold
        self.verbose = verbose

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask=None,
        timestep=None,
        post_patch_height=None,
        post_patch_width=None,
    ):
        batch_size = hidden_states.shape[0]
        if self.residual_diff_threshold <= 0.0 or batch_size > 2:
            if batch_size > 2:
                print("Batch size > 2 (for SANA CFG)" " currently not supported")

            first_transformer_block = self.transformer_blocks[0]
            hidden_states = first_transformer_block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                height=post_patch_height,
                width=post_patch_width,
                skip_first_layer=False,
            )
            return hidden_states

        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]

        hidden_states = first_transformer_block.forward_layer_at(
            0,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            height=post_patch_height,
            width=post_patch_width,
        )
        first_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        can_use_cache, _ = get_can_use_cache(
            first_hidden_states_residual,
            threshold=self.residual_diff_threshold,
            parallelized=self.transformer is not None and getattr(self.transformer, "_is_parallelized", False),
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            del first_hidden_states_residual
            if self.verbose:
                print("Cache hit!!!")
            hidden_states, _ = apply_prev_hidden_states_residual(hidden_states, None)
        else:
            if self.verbose:
                print("Cache miss!!!")
            set_buffer("first_multi_hidden_states_residual", first_hidden_states_residual)
            del first_hidden_states_residual

            hidden_states, hidden_states_residual = self.call_remaining_transformer_blocks(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                post_patch_height=post_patch_height,
                post_patch_width=post_patch_width,
            )
            set_buffer("multi_hidden_states_residual", hidden_states_residual)
        torch._dynamo.graph_break()

        return hidden_states

    def call_remaining_transformer_blocks(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask=None,
        timestep=None,
        post_patch_height=None,
        post_patch_width=None,
    ):
        first_transformer_block = self.transformer_blocks[0]
        original_hidden_states = hidden_states
        hidden_states = first_transformer_block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            height=post_patch_height,
            width=post_patch_width,
            skip_first_layer=True,
        )
        hidden_states_residual = hidden_states - original_hidden_states

        return hidden_states, hidden_states_residual


class FluxCachedTransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module = None,
        use_double_fb_cache: bool = True,
        residual_diff_threshold_multi: float,
        residual_diff_threshold_single: float,
        return_hidden_states_first: bool = True,
        return_hidden_states_only: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer.transformer_blocks
        self.single_transformer_blocks = transformer.single_transformer_blocks

        self.use_double_fb_cache = use_double_fb_cache
        self.residual_diff_threshold_multi = residual_diff_threshold_multi
        self.residual_diff_threshold_single = residual_diff_threshold_single

        self.return_hidden_states_first = return_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only
        self.verbose = verbose

        self.m = self.transformer_blocks[0].m
        self.dtype = torch.bfloat16 if self.m.isBF16() else torch.float16
        self.device = transformer.device

    @staticmethod
    def pack_rotemb(rotemb: torch.Tensor) -> torch.Tensor:
        assert rotemb.dtype == torch.float32
        B = rotemb.shape[0]
        M = rotemb.shape[1]
        D = rotemb.shape[2] * 2
        msg_shape = "rotemb shape must be (B, M, D//2, 1, 2)"
        assert rotemb.shape == (B, M, D // 2, 1, 2), msg_shape
        assert M % 16 == 0
        assert D % 8 == 0
        rotemb = rotemb.reshape(B, M // 16, 16, D // 8, 8)
        rotemb = rotemb.permute(0, 1, 3, 2, 4)
        # 16*8 pack, FP32 accumulator (C) format
        # https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-c
        rotemb = rotemb.reshape(*rotemb.shape[0:3], 2, 8, 4, 2)
        rotemb = rotemb.permute(0, 1, 2, 4, 5, 3, 6)
        rotemb = rotemb.contiguous()
        rotemb = rotemb.view(B, M, D)
        return rotemb

    def update_residual_diff_threshold(
        self, use_double_fb_cache=True, residual_diff_threshold_multi=0.12, residual_diff_threshold_single=0.09
    ):
        self.use_double_fb_cache = use_double_fb_cache
        self.residual_diff_threshold_multi = residual_diff_threshold_multi
        self.residual_diff_threshold_single = residual_diff_threshold_single

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        joint_attention_kwargs=None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
    ):
        batch_size = hidden_states.shape[0]
        txt_tokens = encoder_hidden_states.shape[1]
        img_tokens = hidden_states.shape[1]

        original_dtype = hidden_states.dtype
        original_device = hidden_states.device

        hidden_states = hidden_states.to(self.dtype).to(original_device)
        encoder_hidden_states = encoder_hidden_states.to(self.dtype).to(original_device)
        temb = temb.to(self.dtype).to(original_device)
        image_rotary_emb = image_rotary_emb.to(original_device)

        if controlnet_block_samples is not None:
            controlnet_block_samples = (
                torch.stack(controlnet_block_samples).to(original_device) if len(controlnet_block_samples) > 0 else None
            )
        if controlnet_single_block_samples is not None:
            controlnet_single_block_samples = (
                torch.stack(controlnet_single_block_samples).to(original_device)
                if len(controlnet_single_block_samples) > 0
                else None
            )

        assert image_rotary_emb.ndim == 6
        assert image_rotary_emb.shape[0] == 1
        assert image_rotary_emb.shape[1] == 1
        # [1, tokens, head_dim/2, 1, 2] (sincos)
        total_tokens = txt_tokens + img_tokens
        assert image_rotary_emb.shape[2] == 1 * total_tokens

        image_rotary_emb = image_rotary_emb.reshape([1, txt_tokens + img_tokens, *image_rotary_emb.shape[3:]])
        rotary_emb_txt = image_rotary_emb[:, :txt_tokens, ...]
        rotary_emb_img = image_rotary_emb[:, txt_tokens:, ...]
        rotary_emb_single = image_rotary_emb

        rotary_emb_txt = self.pack_rotemb(pad_tensor(rotary_emb_txt, 256, 1))
        rotary_emb_img = self.pack_rotemb(pad_tensor(rotary_emb_img, 256, 1))
        rotary_emb_single = self.pack_rotemb(pad_tensor(rotary_emb_single, 256, 1))

        if (self.residual_diff_threshold_multi < 0.0):
            if batch_size > 1 and self.verbose:
                print("Batch size > 1 currently not supported")

            hidden_states = self.m.forward(
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_img,
                rotary_emb_txt,
                rotary_emb_single,
                controlnet_block_samples,
                controlnet_single_block_samples,
                skip_first_layer,
            )

            hidden_states = hidden_states.to(original_dtype).to(original_device)

            encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
            hidden_states = hidden_states[:, txt_tokens:, ...]

            if self.return_hidden_states_only:
                return hidden_states
            if self.return_hidden_states_first:
                return hidden_states, encoder_hidden_states
            return encoder_hidden_states, hidden_states

        remaining_kwargs = {
            "temb": temb,
            "rotary_emb_img": rotary_emb_img,
            "rotary_emb_txt": rotary_emb_txt,
            "rotary_emb_single": rotary_emb_single,
            "controlnet_block_samples": controlnet_block_samples,
            "controlnet_single_block_samples": controlnet_single_block_samples,
            "txt_tokens": txt_tokens,
        }
        

        original_hidden_states = hidden_states
        first_hidden_states, first_encoder_hidden_states = self.m.forward_layer(
            0,
            hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb_img,
            rotary_emb_txt,
            controlnet_block_samples,
            controlnet_single_block_samples,
        )
        hidden_states = first_hidden_states
        encoder_hidden_states = first_encoder_hidden_states
        first_hidden_states_residual_multi = hidden_states - original_hidden_states
        del original_hidden_states

        if self.use_double_fb_cache:
            call_remaining_fn = self.call_remaining_multi_transformer_blocks
        else:
            call_remaining_fn = self.call_remaining_FBCache_transformer_blocks

        torch._dynamo.graph_break()
        updated_h, updated_enc, threshold, can_use_cache = check_and_apply_cache(
            first_residual=first_hidden_states_residual_multi,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            threshold=self.residual_diff_threshold_multi,
            parallelized=(self.transformer is not None and getattr(self.transformer, "_is_parallelized", False)),
            mode="multi",
            verbose=self.verbose,
            Force_hit=False,
            call_remaining_fn=call_remaining_fn,
            remaining_kwargs=remaining_kwargs,
        )
        mult_decay_factor = 0.98
        sing_decay_factor = 0.90
        
        #self.residual_diff_threshold_multi = threshold * mult_decay_factor
        if not self.use_double_fb_cache:
            if self.return_hidden_states_only:
                return updated_h
            if self.return_hidden_states_first:
                return updated_h, updated_enc
            return updated_enc, updated_h

        # DoubleFBCache
        cat_hidden_states = torch.cat([updated_enc, updated_h], dim=1)
        original_cat = cat_hidden_states
        cat_hidden_states = self.m.forward_single_layer(0, cat_hidden_states, temb, rotary_emb_single)

        first_hidden_states_residual_single = cat_hidden_states - original_cat
        del original_cat

        call_remaining_fn_single = self.call_remaining_single_transformer_blocks

        updated_cat, _, threshold, _ = check_and_apply_cache(
            first_residual=first_hidden_states_residual_single,
            hidden_states=cat_hidden_states,
            encoder_hidden_states=None,
            threshold=self.residual_diff_threshold_single,
            parallelized=(self.transformer is not None and getattr(self.transformer, "_is_parallelized", False)),
            mode="single",
            verbose=self.verbose,
            call_remaining_fn=call_remaining_fn_single,
            Force_hit=can_use_cache,
            remaining_kwargs=remaining_kwargs,
        )
        #self.residual_diff_threshold_single = threshold * sing_decay_factor
        
        if self.verbose:
            print("MULTI_CACHE_HIT: ",MULT_CACHE_HIT)
            print("SINGLE_CACHE_HIT: ",SING_CACHE_HIT)

        # torch._dynamo.graph_break()

        final_enc = updated_cat[:, :txt_tokens, ...]
        final_h = updated_cat[:, txt_tokens:, ...]

        final_h = final_h.to(original_dtype).to(original_device)
        final_enc = final_enc.to(original_dtype).to(original_device)

        if self.return_hidden_states_only:
            return final_h
        if self.return_hidden_states_first:
            return final_h, final_enc
        return final_enc, final_h

    def call_remaining_FBCache_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=True,
        txt_tokens=None,
    ):
        original_dtype = hidden_states.dtype
        original_device = hidden_states.device
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        hidden_states = self.m.forward(
            hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb_img,
            rotary_emb_txt,
            rotary_emb_single,
            controlnet_block_samples,
            controlnet_single_block_samples,
            skip_first_layer,
        )

        hidden_states = hidden_states.to(original_dtype).to(original_device)

        encoder_hidden_states = hidden_states[:, :txt_tokens, ...]
        hidden_states = hidden_states[:, txt_tokens:, ...]

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        hidden_states_residual = hidden_states - original_hidden_states
        enc_residual = encoder_hidden_states - original_encoder_hidden_states

        return hidden_states, encoder_hidden_states, hidden_states_residual, enc_residual

    def call_remaining_multi_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
        txt_tokens=None,
    ):
        start_idx = 1
        original_hidden_states = hidden_states.clone()
        original_encoder_hidden_states = encoder_hidden_states.clone()

        for idx in range(start_idx, num_transformer_blocks):
            hidden_states, encoder_hidden_states = self.m.forward_layer(
                idx,
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_img,
                rotary_emb_txt,
                controlnet_block_samples,
                controlnet_single_block_samples,
            )

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        hs_res = hidden_states - original_hidden_states
        enc_res = encoder_hidden_states - original_encoder_hidden_states
        return hidden_states, encoder_hidden_states, hs_res, enc_res

    def call_remaining_single_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb_img: torch.Tensor,
        rotary_emb_txt: torch.Tensor,
        rotary_emb_single: torch.Tensor,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        skip_first_layer=False,
        txt_tokens=None,
    ):
        start_idx = 1
        original_hidden_states = hidden_states.clone()

        for idx in range(start_idx, num_single_transformer_blocks):
            hidden_states = self.m.forward_single_layer(
                idx,
                hidden_states,
                temb,
                rotary_emb_single,
            )

        hidden_states = hidden_states.contiguous()
        hs_res = hidden_states - original_hidden_states
        return hidden_states, hs_res
