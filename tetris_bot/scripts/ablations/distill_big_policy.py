"""Distill MCTS replay data into a large policy network and play it with no search.

Stages (``--mode``):
- ``prepare``: convert replay ``training_data.npz`` files into compact ``.npy`` caches
  (top-48 sparse policy targets, bit-packed masks, fp16 aux).
- ``train``: train a policy network on the GPU-resident cache and play policy-only
  (argmax, zero simulations) games on fixed seeds during and after training. Two
  architectures: ``transformer`` (one token per board cell plus piece/context tokens,
  per-cell 4-rotation placement logits) and ``resnet`` (global-pooling ResNet with a
  spatial ``4x20x10`` policy head). Everything is logged to W&B as it trains.
- ``eval``: policy-only games for a saved checkpoint, or for an ONNX incumbent via the
  Rust predictor (the baseline the attack-vs-search-depth chart used).
- ``backfill``: upload an existing run's ``metrics.jsonl`` to W&B.

The networks here never run inside MCTS, so there is no board-embedding cache, split
export or Rust inference path to stay compatible with.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from simple_parsing import parse

from tetris_bot.action_space import ACTION_TO_CANONICAL_FLAT_INDEX
from tetris_bot.constants import BENCHMARKS_DIR, NUM_ACTIONS, QUEUE_SIZE
from tetris_bot.ml.aux_features import AUX_FEATURE_LAYOUT
from tetris_bot.ml.policy_mirroring import mirror_training_tensors
from tetris_bot.scripts.ablations.compare_offline_architectures import (
    build_aux_batch_from_npz,
)
from tetris_core.tetris_core import (
    TetrisEnv,
    debug_encode_state,
    debug_get_action_mask,
    debug_predict_masked_from_tensors,
)

logger = structlog.get_logger()

BOARD_H, BOARD_W = 20, 10
NUM_CELLS = BOARD_H * BOARD_W
AUX_DIM = 80
POLICY_TOPK = 48
VALUE_SCALE = 20.0
CACHE_FIELDS = (
    "boards",
    "aux",
    "policy_idx",
    "policy_val",
    "mask_bits",
    "values",
    "games",
)
# Policy-only (0 simulation) attack of the deterministic-phase run's networks, measured
# with `--mode eval --baseline_onnx` (2026-11-11). `incumbent` is checkpoints/incumbent.onnx
# (step 3,332,187), the 47.1 point in docs/attack_vs_search_depth.png; `final_sync` is
# final_weights/sync_step_3505140.onnx, the run's last and best no-search policy.
BASELINES = {
    "incumbent_seeds0_19": 47.1,
    "incumbent_seeds0_199": 49.09,
    "final_sync_seeds0_19": 52.85,
    "final_sync_seeds0_199": 50.265,
    "mcts100_seeds0_19": 54.4,
}


@dataclass
class ScriptArgs:
    mode: str = "train"  # prepare | train | eval | backfill
    data_paths: list[str] = field(
        default_factory=list
    )  # npz (prepare) or cache dirs (train)
    cache_root: Path = BENCHMARKS_DIR / "big_policy" / "cache"
    output_dir: Path = BENCHMARKS_DIR / "big_policy" / "runs"
    run_name: str = ""
    arch: str = "transformer"  # transformer | resnet
    # transformer
    d_model: int = 384
    layers: int = 12
    heads: int = 6
    mlp_hidden: int = 1024
    conv_stem_blocks: int = 4  # 0 = pure transformer over raw cells
    conv_stem_channels: int = 128
    # resnet
    channels: int = 128
    blocks: int = 10
    cond_dim: int = 256
    head_channels: int = 64
    # optimisation
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    train_minutes: float = (
        120.0  # LR schedule runs on train-time, so every budget ends fully decayed
    )
    value_loss_weight: float = 0.25
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    mirror_probability: float = (
        0.5  # fraction of each batch that is left-right mirrored
    )
    compile: bool = True
    seed: int = 0
    # logging / eval cadence (train-time seconds)
    log_every_seconds: float = 10.0
    val_every_seconds: float = 120.0
    game_eval_every_minutes: float = 10.0
    val_game_mod: int = 50  # games with game_number % mod == 0 are held out
    # policy-only game eval
    eval_seeds: int = 200
    final_eval_seeds: int = 1000
    max_placements: int = 50
    checkpoint: Path | None = None  # eval mode: a saved .pt from train mode
    baseline_onnx: str = (
        ""  # eval mode: an incumbent .onnx played through the Rust predictor
    )
    eval_workers: int = 32
    wandb_project: str = "tetris-no-search"  # empty disables W&B


# ---------------------------------------------------------------- data


def prepare_cache(npz_path: Path, out_dir: Path, game_offset: int) -> None:
    torch.set_num_threads(16)
    data = np.load(npz_path)
    n = data["boards"].shape[0]
    log = logger.bind(source=str(npz_path), examples=n)
    log.info("Preparing cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        out_dir / "boards.npy",
        data["boards"].reshape(n, BOARD_H, BOARD_W).astype(np.uint8),
    )
    aux = build_aux_batch_from_npz(data, np.arange(n))
    if aux.shape[1] != AUX_DIM:
        raise ValueError(f"aux width {aux.shape[1]} != {AUX_DIM}")
    np.save(out_dir / "aux.npy", aux.astype(np.float16))
    del aux
    np.save(out_dir / "values.npy", data["value_targets"].astype(np.float32))
    np.save(out_dir / "games.npy", data["game_numbers"].astype(np.int64) + game_offset)
    masks = data["action_masks"]
    if masks.shape[1] != NUM_ACTIONS:
        raise ValueError(f"mask width {masks.shape[1]} != {NUM_ACTIONS}")
    np.save(out_dir / "mask_bits.npy", np.packbits(masks, axis=1, bitorder="little"))
    del masks
    policy = torch.from_numpy(data["policy_targets"])
    idx_out = np.empty((n, POLICY_TOPK), dtype=np.int16)
    val_out = np.empty((n, POLICY_TOPK), dtype=np.float16)
    lost_mass = 0.0
    for start in range(0, n, 250_000):
        chunk = policy[start : start + 250_000]
        vals, idxs = chunk.topk(POLICY_TOPK, dim=1)
        lost_mass = max(lost_mass, float((chunk.sum(1) - vals.sum(1)).max()))
        idx_out[start : start + len(chunk)] = idxs.numpy().astype(np.int16)
        val_out[start : start + len(chunk)] = vals.numpy().astype(np.float16)
    np.save(out_dir / "policy_idx.npy", idx_out)
    np.save(out_dir / "policy_val.npy", val_out)
    log.info("Prepared cache", out_dir=str(out_dir), max_policy_mass_dropped=lost_mass)


def run_prepare(args: ScriptArgs) -> None:
    jobs = []
    for i, raw in enumerate(args.data_paths):
        path = Path(raw)
        name = f"{path.parent.name}__{path.stem}"
        jobs.append((path, args.cache_root / name, i * 10**9))
    with Pool(len(jobs)) as pool:
        pool.starmap(prepare_cache, jobs)


@dataclass
class GpuDataset:
    boards: torch.Tensor
    aux: torch.Tensor
    policy_idx: torch.Tensor
    policy_val: torch.Tensor
    mask_bits: torch.Tensor
    values: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor


def load_dataset(
    cache_dirs: list[str], val_game_mod: int, device: torch.device
) -> GpuDataset:
    parts: dict[str, list[np.ndarray]] = {name: [] for name in CACHE_FIELDS}
    for raw in cache_dirs:
        for name in CACHE_FIELDS:
            parts[name].append(np.load(Path(raw) / f"{name}.npy"))
    arrays = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    games = arrays.pop("games")
    is_val = games % val_game_mod == 0
    tensors = {name: torch.from_numpy(arr).to(device) for name, arr in arrays.items()}
    dataset = GpuDataset(
        **tensors,
        train_idx=torch.from_numpy(np.flatnonzero(~is_val)).to(device),
        val_idx=torch.from_numpy(np.flatnonzero(is_val)).to(device),
    )
    logger.info(
        "Loaded dataset",
        examples=len(games),
        train=len(dataset.train_idx),
        val=len(dataset.val_idx),
        gpu_gb=round(
            sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9, 2
        ),
    )
    return dataset


_MASK_SHIFTS: dict[torch.device, torch.Tensor] = {}


def unpack_masks(bits: torch.Tensor) -> torch.Tensor:
    shifts = _MASK_SHIFTS.setdefault(
        bits.device, torch.arange(8, device=bits.device, dtype=torch.uint8)
    )
    return (
        ((bits.unsqueeze(-1) >> shifts) & 1).bool().reshape(bits.shape[0], NUM_ACTIONS)
    )


def gather_batch(
    ds: GpuDataset, rows: torch.Tensor, mirror_rows: int = 0
) -> tuple[torch.Tensor, ...]:
    boards = ds.boards[rows].unsqueeze(1).float()
    aux = ds.aux[rows].float()
    policy = torch.zeros(len(rows), NUM_ACTIONS, device=rows.device)
    policy.scatter_add_(1, ds.policy_idx[rows].long(), ds.policy_val[rows].float())
    policy = policy / policy.sum(1, keepdim=True)
    masks = unpack_masks(ds.mask_bits[rows])
    if mirror_rows > 0:
        mirrored = mirror_training_tensors(
            boards[:mirror_rows],
            aux[:mirror_rows],
            policy[:mirror_rows],
            masks[:mirror_rows],
        )
        boards, aux, policy, masks = (
            torch.cat([m, t[mirror_rows:]])
            for m, t in zip(mirrored, (boards, aux, policy, masks), strict=True)
        )
    return boards, aux, policy, masks, ds.values[rows] / VALUE_SCALE


# ---------------------------------------------------------------- transformer


class TransformerBlock(nn.Module):
    """Pre-norm block: QK-normed multi-head attention and a SwiGLU MLP."""

    def __init__(self, d_model: int, heads: int, mlp_hidden: int):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by heads {heads}")
        self.heads = heads
        head_dim = d_model // heads
        self.attn_norm = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp_in = nn.Linear(d_model, 2 * mlp_hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, d_model, bias=False)
        # Every block starts as the identity, so depth does not destabilise early training.
        nn.init.zeros_(self.attn_out.weight)
        nn.init.zeros_(self.mlp_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        q, k, v = (
            self.qkv(self.attn_norm(x))
            .view(batch, tokens, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v)
        x = x + self.attn_out(attended.transpose(1, 2).reshape(batch, tokens, width))
        gate, up = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.mlp_out(F.silu(gate) * up)


def _slice_width(s: slice) -> int:
    return s.stop - s.start


class ConvStemBlock(nn.Module):
    """Pre-norm residual 3x3 conv block; gives cell tokens a local-neighbourhood prior."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(h)))


class TransformerPolicyNet(nn.Module):
    """One token per board cell plus queue, current/hold piece, scalar, value and hold tokens.

    With ``conv_stem_blocks > 0`` the cell tokens come from a residual 3x3 conv stack
    over the board (local patterns: fits, slots, overhangs) and the transformer then
    mixes them globally; with 0 each cell token is just filled/empty plus its position.
    Placement logits come from the cell tokens (4 rotations per cell, read at the
    canonical cell each action index maps to); the hold logit and value come from their
    own learned query tokens.
    """

    def __init__(
        self,
        d_model: int,
        layers: int,
        heads: int,
        mlp_hidden: int,
        conv_stem_blocks: int,
        conv_stem_channels: int,
    ):
        super().__init__()
        layout = AUX_FEATURE_LAYOUT
        self.current_slice = layout.current_piece
        self.hold_slice = layout.hold_piece
        self.queue_slice = layout.next_queue
        self.register_buffer("aux_mean", torch.zeros(AUX_DIM))
        self.register_buffer("aux_std", torch.ones(AUX_DIM))

        self.cell_filled = nn.Parameter(torch.zeros(d_model))
        self.conv_stem: nn.Module | None = None
        if conv_stem_blocks > 0:
            self.conv_stem = nn.Sequential(
                nn.Conv2d(1, conv_stem_channels, 3, padding=1),
                *(ConvStemBlock(conv_stem_channels) for _ in range(conv_stem_blocks)),
                nn.GroupNorm(8, conv_stem_channels),
                nn.SiLU(),
                nn.Conv2d(conv_stem_channels, d_model, 1),
            )
        self.cell_pos = nn.Parameter(torch.zeros(NUM_CELLS, d_model))
        self.current_embed = nn.Linear(_slice_width(layout.current_piece), d_model)
        self.hold_embed = nn.Linear(_slice_width(layout.hold_piece), d_model)
        self.queue_embed = nn.Linear(
            _slice_width(layout.next_queue) // QUEUE_SIZE, d_model
        )
        self.queue_pos = nn.Parameter(torch.zeros(QUEUE_SIZE, d_model))
        self.scalar_embed = nn.Sequential(
            nn.Linear(AUX_DIM, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.value_query = nn.Parameter(torch.zeros(d_model))
        self.hold_query = nn.Parameter(torch.zeros(d_model))
        for param in (
            self.cell_filled,
            self.cell_pos,
            self.queue_pos,
            self.value_query,
            self.hold_query,
        ):
            nn.init.normal_(param, std=0.02)

        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, heads, mlp_hidden) for _ in range(layers)
        )
        self.final_norm = nn.RMSNorm(d_model)
        self.placement_head = nn.Linear(d_model, 4)
        self.hold_head = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1)
        )
        self.register_buffer(
            "canonical_index",
            torch.from_numpy(ACTION_TO_CANONICAL_FLAT_INDEX.astype(np.int64)),
            persistent=False,
        )

    def forward(
        self, board: torch.Tensor, aux: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = board.shape[0]
        cells = board.reshape(batch, NUM_CELLS, 1) * self.cell_filled + self.cell_pos
        if self.conv_stem is not None:
            # Walls are implicit in the zero padding, which reads as empty; the filled
            # embedding and positions above still carry absolute location.
            cells = cells + self.conv_stem(board).flatten(2).transpose(1, 2)
        queue = aux[:, self.queue_slice].reshape(batch, QUEUE_SIZE, -1)
        context = torch.stack(
            [
                self.current_embed(aux[:, self.current_slice]),
                self.hold_embed(aux[:, self.hold_slice]),
                self.scalar_embed((aux - self.aux_mean) / self.aux_std),
                self.value_query.expand(batch, -1),
                self.hold_query.expand(batch, -1),
            ],
            dim=1,
        )
        x = torch.cat([cells, self.queue_embed(queue) + self.queue_pos, context], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # (batch, cell, rotation) -> rotation-major grid, matching placement_grid_flat_index.
        grid = self.placement_head(x[:, :NUM_CELLS]).transpose(1, 2).reshape(batch, -1)
        logits = torch.cat(
            [grid[:, self.canonical_index], self.hold_head(x[:, -1])], dim=1
        )
        return logits, self.value_head(x[:, -2]).squeeze(1)


# ---------------------------------------------------------------- resnet


class GlobalBiasBlock(nn.Module):
    """Residual block whose middle activation gets a bias from the pooled board and the piece context."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.global_fc = nn.Linear(2 * channels + cond_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3)), cond], dim=1)
        h = F.relu(h + self.global_fc(pooled)[:, :, None, None])
        return F.relu(x + self.bn2(self.conv2(h)))


class BigPolicyNet(nn.Module):
    def __init__(self, channels: int, blocks: int, cond_dim: int, head_channels: int):
        super().__init__()
        self.register_buffer("aux_mean", torch.zeros(AUX_DIM))
        self.register_buffer("aux_std", torch.ones(AUX_DIM))
        self.cond = nn.Sequential(
            nn.Linear(AUX_DIM, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
        )
        self.stem = nn.Conv2d(2, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.stem_cond = nn.Linear(cond_dim, channels)
        self.blocks = nn.ModuleList(
            GlobalBiasBlock(channels, cond_dim) for _ in range(blocks)
        )
        self.policy_conv = nn.Conv2d(channels, head_channels, 3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(head_channels)
        self.policy_out = nn.Conv2d(head_channels, 4, 1)
        pooled = 2 * channels + cond_dim
        self.hold_head = nn.Sequential(
            nn.Linear(pooled, 256), nn.SiLU(), nn.Linear(256, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(pooled, 256), nn.SiLU(), nn.Linear(256, 1)
        )
        self.register_buffer(
            "canonical_index",
            torch.from_numpy(ACTION_TO_CANONICAL_FLAT_INDEX.astype(np.int64)),
            persistent=False,
        )

    def forward(
        self, board: torch.Tensor, aux: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self.cond((aux - self.aux_mean) / self.aux_std)
        x = torch.cat([board, torch.ones_like(board)], dim=1)
        x = F.relu(self.stem_bn(self.stem(x)) + self.stem_cond(cond)[:, :, None, None])
        for block in self.blocks:
            x = block(x, cond)
        grid = self.policy_out(F.relu(self.policy_bn(self.policy_conv(x))))
        placement_logits = grid.flatten(1)[:, self.canonical_index]
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3)), cond], dim=1)
        logits = torch.cat([placement_logits, self.hold_head(pooled)], dim=1)
        return logits, self.value_head(pooled).squeeze(1)


def build_model(config: dict) -> nn.Module:
    if config["arch"] == "transformer":
        return TransformerPolicyNet(
            config["d_model"],
            config["layers"],
            config["heads"],
            config["mlp_hidden"],
            config.get("conv_stem_blocks", 0),
            config.get("conv_stem_channels", 128),
        )
    if config["arch"] == "resnet":
        return BigPolicyNet(
            config["channels"],
            config["blocks"],
            config["cond_dim"],
            config["head_channels"],
        )
    raise ValueError(f"unknown arch {config['arch']!r}")


# ---------------------------------------------------------------- policy-only games


@torch.no_grad()
def play_policy_games(
    model: nn.Module, seeds: list[int], max_placements: int, device: torch.device
) -> list[int]:
    """All seeds play in lockstep so each move is one batched forward pass."""
    model.eval()
    envs = [TetrisEnv.with_seed(BOARD_W, BOARD_H, seed) for seed in seeds]
    active = list(range(len(envs)))
    while active:
        encoded = [debug_encode_state(envs[i], max_placements) for i in active]
        boards = torch.tensor(
            np.array([b for b, _ in encoded], dtype=np.float32), device=device
        )
        aux = torch.tensor(
            np.array([a for _, a in encoded], dtype=np.float32), device=device
        )
        masks = torch.tensor(
            np.array([debug_get_action_mask(envs[i]) for i in active]), device=device
        )
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits, _ = model(boards.view(-1, 1, BOARD_H, BOARD_W), aux)
        actions = logits.float().masked_fill(~masks, float("-inf")).argmax(1).tolist()
        for i, action in zip(active, actions):
            if envs[i].execute_action_index(action) is None:
                raise RuntimeError(f"illegal action {action} for seed {seeds[i]}")
        active = [
            i
            for i in active
            if envs[i].placement_count < max_placements and not envs[i].game_over
        ]
    return [int(env.attack) for env in envs]


def _play_onnx_game(job: tuple[str, int, int]) -> int:
    model_path, seed, max_placements = job
    env = TetrisEnv.with_seed(BOARD_W, BOARD_H, seed)
    while env.placement_count < max_placements and not env.game_over:
        board, aux = debug_encode_state(env, max_placements)
        mask = debug_get_action_mask(env)
        policy, _ = debug_predict_masked_from_tensors(model_path, board, aux, mask)
        action = int(np.argmax(np.where(mask, policy, -1.0)))
        if env.execute_action_index(action) is None:
            raise RuntimeError(f"illegal action {action} for seed {seed}")
    return int(env.attack)


def summarize_attacks(attacks: list[int]) -> dict[str, float]:
    arr = np.asarray(attacks, dtype=np.float64)
    p10, p50, p90 = np.percentile(arr, [10, 50, 90])
    return {
        "attack_seeds0_19": float(arr[:20].mean()),
        "attack_all": float(arr.mean()),
        "attack_all_sem": float(arr.std(ddof=1) / math.sqrt(len(arr))),
        "attack_p10": float(p10),
        "attack_p50": float(p50),
        "attack_p90": float(p90),
        "games": len(arr),
    }


# ---------------------------------------------------------------- training


def lr_at(args: ScriptArgs, step: int, progress: float) -> float:
    warmup = min(1.0, (step + 1) / args.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return args.learning_rate * warmup * (0.02 + 0.98 * cosine)


def compute_losses(
    model: nn.Module, batch: tuple[torch.Tensor, ...]
) -> dict[str, torch.Tensor]:
    boards, aux, policy, masks, values = batch
    logits, value_pred = model(boards, aux)
    logits = logits.float().masked_fill(~masks, -1e9)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(policy * log_probs).sum(1).mean()
    # Cross-entropy minus target entropy: 0 means the policy matches the MCTS target exactly.
    target_entropy = -(policy * torch.log(policy.clamp_min(1e-12))).sum(1).mean()
    value_pred = value_pred.float()
    value_loss = F.mse_loss(value_pred, values)
    # 1 = value explains all variance in the remaining-attack target, 0 = no better than its mean.
    value_explained_variance = 1.0 - (
        value_pred - values
    ).var() / values.var().clamp_min(1e-8)
    top1 = (logits.argmax(1) == policy.argmax(1)).float().mean()
    return {
        "policy_loss": policy_loss,
        "policy_kl": policy_loss - target_entropy,
        "value_loss": value_loss,
        "value_explained_variance": value_explained_variance,
        # Typical miss in attack points (targets are remaining attack / VALUE_SCALE).
        "value_rmse_attack": value_loss.sqrt() * VALUE_SCALE,
        "top1": top1,
    }


@torch.no_grad()
def validate(
    model: nn.Module, ds: GpuDataset, max_rows: int = 200_000
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    rows_all = ds.val_idx[:max_rows]
    for rows in rows_all.split(8192):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=rows.is_cuda):
            losses = compute_losses(model, gather_batch(ds, rows))
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * len(rows)
    return {f"val_{k}": v / len(rows_all) for k, v in totals.items()}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_run_name(args: ScriptArgs) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.arch == "transformer":
        return (
            f"transformer_d{args.d_model}l{args.layers}c{args.conv_stem_blocks}_{stamp}"
        )
    return f"resnet_c{args.channels}b{args.blocks}_{stamp}"


def run_train(args: ScriptArgs) -> None:
    torch.manual_seed(args.seed)
    device = pick_device()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_name = args.run_name or default_run_name(args)
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(asdict(args), default=str, indent=1))
    metrics_file = (out_dir / "metrics.jsonl").open("a")

    ds = load_dataset(args.data_paths, args.val_game_mod, device)
    model = build_model(asdict(args)).to(device)
    sample = ds.aux[
        ds.train_idx[torch.randperm(len(ds.train_idx), device=device)[:500_000]]
    ].float()
    model.aux_mean.copy_(sample.mean(0))
    model.aux_std.copy_(sample.std(0).clamp_min(1e-3))
    if args.arch == "resnet":
        model = model.to(memory_format=torch.channels_last)
    ema_model = copy.deepcopy(model).eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    params = sum(p.numel() for p in model.parameters())
    gpu_name = torch.cuda.get_device_name() if device.type == "cuda" else str(device)
    logger.info("Model", run=run_name, params_m=round(params / 1e6, 2), device=gpu_name)
    wandb_run = init_wandb(args, run_name, out_dir, params=params, gpu=gpu_name, ds=ds)

    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    model_params = list(model.parameters())
    ema_params = list(ema_model.parameters())

    def sync_ema_buffers() -> None:
        # The EMA averages parameters only; BatchNorm running stats must come from the live model.
        for ema_buffer, buffer in zip(
            ema_model.buffers(), model.buffers(), strict=True
        ):
            ema_buffer.copy_(buffer)

    train_step = compute_losses
    if args.compile and device.type == "cuda":
        train_step = torch.compile(compute_losses)

    seeds = list(range(args.eval_seeds))
    budget = args.train_minutes * 60.0
    train_examples = len(ds.train_idx)
    mirror_rows = int(round(args.batch_size * args.mirror_probability))
    train_time, step, samples = 0.0, 0, 0
    last_log = last_val = last_game_eval = 0.0
    best_attack = -math.inf
    window: dict[str, torch.Tensor] = {}
    window_steps, window_start = 0, time.perf_counter()
    run_start = time.perf_counter()

    def record(row: dict) -> None:
        row.update(
            step=step,
            samples=samples,
            epoch=samples / train_examples,
            train_minutes=round(train_time / 60, 3),
            wall_minutes=round((time.perf_counter() - run_start) / 60, 3),
        )
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        if wandb_run is not None:
            wandb_run.log(wandb_row(row), step=step)
        logger.info(
            "metrics",
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()},
        )

    def save(path: Path, extra: dict) -> None:
        sync_ema_buffers()
        torch.save(
            {
                "model_state_dict": ema_model.state_dict(),
                "args": json.loads(json.dumps(asdict(args), default=str)),
                **extra,
            },
            path,
        )

    def game_eval(eval_seeds: list[int]) -> dict:
        sync_ema_buffers()
        start = time.perf_counter()
        attacks = play_policy_games(ema_model, eval_seeds, args.max_placements, device)
        result: dict = summarize_attacks(attacks)
        result["game_eval_seconds"] = time.perf_counter() - start
        if wandb_run is not None:
            wandb_run.log({"game/attack_hist": wandb.Histogram(attacks)}, step=step)
        return result

    model.train()
    while train_time < budget:
        tick = time.perf_counter()
        for group in optimizer.param_groups:
            group["lr"] = lr_at(args, step, train_time / budget)
        rows = ds.train_idx[
            torch.randint(train_examples, (args.batch_size,), device=device)
        ]
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            losses = train_step(model, gather_batch(ds, rows, mirror_rows))
        loss = losses["policy_loss"] + args.value_loss_weight * losses["value_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        losses["grad_norm"] = torch.nn.utils.clip_grad_norm_(
            model_params, args.grad_clip
        )
        optimizer.step()
        # Warmed-up decay so early evals reflect the model rather than its random init.
        ema_decay = min(args.ema_decay, (1.0 + step) / (10.0 + step))
        torch._foreach_lerp_(ema_params, model_params, 1.0 - ema_decay)
        step += 1
        samples += args.batch_size
        for key, value in losses.items():
            window[key] = window.get(key, 0.0) + value.detach().float()
        window_steps += 1
        if device.type == "cuda" and step % 50 == 0:
            torch.cuda.synchronize()
        train_time += time.perf_counter() - tick

        if train_time - last_log < args.log_every_seconds:
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - window_start
        row: dict = {k: float(v) / window_steps for k, v in window.items()}
        row.update(
            lr=optimizer.param_groups[0]["lr"],
            steps_per_s=window_steps / elapsed,
            samples_per_s=window_steps * args.batch_size / elapsed,
        )
        if device.type == "cuda":
            row["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        if train_time - last_val >= args.val_every_seconds:
            sync_ema_buffers()
            row.update(validate(ema_model, ds))
            last_val = train_time
        if train_time - last_game_eval >= args.game_eval_every_minutes * 60:
            row.update(game_eval(seeds))
            last_game_eval = train_time
            save(out_dir / "latest.pt", {"step": step})
            if row["attack_all"] > best_attack:
                best_attack = row["attack_all"]
                save(out_dir / "best.pt", {"step": step, "attack_all": best_attack})
            row["best_attack_all"] = best_attack
        record(row)
        last_log, window, window_steps, window_start = (
            train_time,
            {},
            0,
            time.perf_counter(),
        )

    save(out_dir / "final.pt", {"step": step})
    sync_ema_buffers()
    final = {**validate(ema_model, ds), **game_eval(list(range(args.final_eval_seeds)))}
    final["final"] = True
    record(final)
    (out_dir / "final.json").write_text(json.dumps(final, indent=1))
    logger.info("Saved", out_dir=str(out_dir))
    if wandb_run is not None:
        for key, value in final.items():
            if isinstance(value, (int, float)):
                wandb_run.summary[f"final/{key}"] = value
        artifact = wandb.Artifact(f"no-search-{run_name}", type="model")
        artifact.add_file(str(out_dir / "final.pt"))
        if (out_dir / "best.pt").exists():
            artifact.add_file(str(out_dir / "best.pt"))
        artifact.add_file(str(out_dir / "metrics.jsonl"))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()


def init_wandb(
    args: ScriptArgs,
    run_name: str,
    out_dir: Path,
    *,
    params: int,
    gpu: str,
    ds: GpuDataset,
):
    if not args.wandb_project:
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            **asdict(args),
            "params": params,
            "params_m": round(params / 1e6, 2),
            "gpu": gpu,
            "train_examples": len(ds.train_idx),
            "val_examples": len(ds.val_idx),
            "baselines": BASELINES,
        },
        dir=str(out_dir),
    )
    for key, value in BASELINES.items():
        run.summary[f"baseline/{key}"] = value
    return run


def wandb_row(row: dict) -> dict:
    """Group metrics so W&B shows game attack, val and train curves as separate panels."""
    out = {}
    for key, value in row.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if key.startswith(("attack", "game", "best_")):
            out[f"game/{key}"] = value
        elif key.startswith("val_"):
            out[f"val/{key[4:]}"] = value
        elif key in ("step", "samples", "epoch", "train_minutes", "wall_minutes"):
            out[f"progress/{key}"] = value
        else:
            out[f"train/{key}"] = value
    if "attack_all" in row:
        # Logged beside the model's own attack so the bar to beat sits on the same chart.
        out["game/baseline_final_sync_seeds0_199"] = BASELINES["final_sync_seeds0_199"]
        out["game/baseline_incumbent_seeds0_19"] = BASELINES["incumbent_seeds0_19"]
    return out


def run_backfill(args: ScriptArgs) -> None:
    run_dir = args.output_dir / args.run_name
    saved_args = json.loads((run_dir / "args.json").read_text())
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=saved_args,
        dir=str(run_dir),
    )
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        run.log(wandb_row(row), step=row["step"])
    run.finish()


def run_eval(args: ScriptArgs) -> None:
    seeds = list(range(args.eval_seeds))
    if args.baseline_onnx:
        with Pool(args.eval_workers) as pool:
            attacks = pool.map(
                _play_onnx_game,
                [(args.baseline_onnx, s, args.max_placements) for s in seeds],
            )
        label = args.baseline_onnx
    else:
        if args.checkpoint is None:
            raise ValueError("eval mode needs --checkpoint or --baseline_onnx")
        device = pick_device()
        # Our own checkpoints; older ones pickled Path objects in `args`.
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Checkpoints from before the transformer was added carry no `arch` field.
        model = build_model({"arch": "resnet", **saved["args"]})
        model.load_state_dict(saved["model_state_dict"])
        attacks = play_policy_games(
            model.to(device), seeds, args.max_placements, device
        )
        label = str(args.checkpoint)
    logger.info(
        "Policy-only eval",
        model=label,
        attacks=attacks[:20],
        **summarize_attacks(attacks),
    )


def main(args: ScriptArgs) -> None:
    modes = {
        "prepare": run_prepare,
        "train": run_train,
        "eval": run_eval,
        "backfill": run_backfill,
    }
    modes[args.mode](args)


if __name__ == "__main__":
    main(parse(ScriptArgs))
