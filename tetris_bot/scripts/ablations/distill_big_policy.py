"""Distill MCTS replay data into a large policy network and play it with no search.

Stages (``--mode``):
- ``prepare``: convert replay ``training_data.npz`` files into compact ``.npy`` caches
  (top-48 sparse policy targets, bit-packed masks, fp16 aux).
- ``train``: train a global-pooling ResNet with a spatial ``4x20x10`` policy head on
  the GPU-resident cache, evaluating policy-only (argmax, zero simulations) games
  on fixed seeds during and after training.
- ``eval``: policy-only games for a saved checkpoint, or for an ONNX incumbent via the
  Rust predictor (the baseline the attack-vs-search-depth chart used).
"""

from __future__ import annotations

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
from tetris_bot.constants import BENCHMARKS_DIR, NUM_ACTIONS
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
AUX_DIM = 80
POLICY_TOPK = 48
MASK_BYTES = NUM_ACTIONS // 8
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


@dataclass
class ScriptArgs:
    mode: str = "train"  # prepare | train | eval | backfill (upload metrics.jsonl of --run_name)
    data_paths: list[str] = field(
        default_factory=list
    )  # npz (prepare) or cache dirs (train)
    cache_root: Path = BENCHMARKS_DIR / "big_policy" / "cache"
    output_dir: Path = BENCHMARKS_DIR / "big_policy" / "runs"
    run_name: str = ""
    # model
    channels: int = 128
    blocks: int = 10
    cond_dim: int = 256
    head_channels: int = 64
    # optimisation
    batch_size: int = 2048
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 300
    train_minutes: float = (
        10.0  # LR schedule runs on train-time, so every budget ends fully decayed
    )
    value_loss_weight: float = 0.25
    val_every_seconds: float = 60.0
    game_eval_every_minutes: float = 5.0
    val_game_mod: int = 50  # games with game_number % mod == 0 are held out
    compile: bool = True
    seed: int = 0
    # policy-only game eval
    eval_seeds: int = 200
    max_placements: int = 50
    checkpoint: Path | None = None  # eval mode: a saved .pt from train mode
    baseline_onnx: str = (
        ""  # eval mode: an incumbent .onnx played through the Rust predictor
    )
    eval_workers: int = 32
    wandb_project: str = "tetris-big-policy"  # empty disables W&B


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


def gather_batch(ds: GpuDataset, rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
    boards = ds.boards[rows].unsqueeze(1).float()
    aux = ds.aux[rows].float()
    policy = torch.zeros(len(rows), NUM_ACTIONS, device=rows.device)
    policy.scatter_add_(1, ds.policy_idx[rows].long(), ds.policy_val[rows].float())
    policy = policy / policy.sum(1, keepdim=True)
    return (
        boards,
        aux,
        policy,
        unpack_masks(ds.mask_bits[rows]),
        ds.values[rows] / VALUE_SCALE,
    )


# ---------------------------------------------------------------- model


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


def build_model(args: ScriptArgs) -> BigPolicyNet:
    return BigPolicyNet(args.channels, args.blocks, args.cond_dim, args.head_channels)


# ---------------------------------------------------------------- policy-only games


@torch.no_grad()
def play_policy_games(
    model: nn.Module, seeds: list[int], max_placements: int, device: torch.device
) -> list[int]:
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
    return {
        "attack_seeds0_19": float(arr[:20].mean()),
        "attack_all": float(arr.mean()),
        "attack_all_sem": float(arr.std(ddof=1) / math.sqrt(len(arr))),
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
    value_loss = F.mse_loss(value_pred.float(), values)
    top1 = (logits.argmax(1) == policy.argmax(1)).float().mean()
    return {"policy_loss": policy_loss, "value_loss": value_loss, "top1": top1}


@torch.no_grad()
def validate(
    model: nn.Module, ds: GpuDataset, max_rows: int = 200_000
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    rows_all = ds.val_idx[:max_rows]
    chunks = rows_all.split(8192)
    for rows in chunks:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=rows.is_cuda):
            losses = compute_losses(model, gather_batch(ds, rows))
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * len(rows)
    model.train()
    return {f"val_{k}": v / len(rows_all) for k, v in totals.items()}


def run_train(args: ScriptArgs) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_name = (
        args.run_name
        or f"c{args.channels}b{args.blocks}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(asdict(args), default=str, indent=1))
    metrics_file = (out_dir / "metrics.jsonl").open("a")
    wandb_run = init_wandb(args, run_name, out_dir)

    ds = load_dataset(args.data_paths, args.val_game_mod, device)
    model = build_model(args).to(device)
    sample = ds.aux[
        ds.train_idx[torch.randperm(len(ds.train_idx), device=device)[:500_000]]
    ].float()
    model.aux_mean.copy_(sample.mean(0))
    model.aux_std.copy_(sample.std(0).clamp_min(1e-3))
    model = model.to(memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model", run=run_name, params_m=round(params / 1e6, 2), device=str(device)
    )
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
    train_step = compute_losses
    if args.compile and device.type == "cuda":
        train_step = torch.compile(compute_losses)

    seeds = list(range(args.eval_seeds))
    budget = args.train_minutes * 60.0
    train_time, step, samples = 0.0, 0, 0
    last_val = last_game_eval = 0.0
    window: dict[str, float] = {}
    window_steps, window_start = 0, time.perf_counter()

    def record(row: dict) -> None:
        row.update(step=step, samples=samples, train_minutes=round(train_time / 60, 2))
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        if wandb_run is not None:
            wandb_run.log(wandb_row(row), step=step)
        logger.info(
            "metrics",
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()},
        )

    def game_eval() -> dict[str, float]:
        start = time.perf_counter()
        result = summarize_attacks(
            play_policy_games(model, seeds, args.max_placements, device)
        )
        model.train()
        result["game_eval_seconds"] = time.perf_counter() - start
        return result

    model.train()
    while train_time < budget:
        tick = time.perf_counter()
        for group in optimizer.param_groups:
            group["lr"] = lr_at(args, step, train_time / budget)
        rows = ds.train_idx[
            torch.randint(len(ds.train_idx), (args.batch_size,), device=device)
        ]
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            losses = train_step(model, gather_batch(ds, rows))
        loss = losses["policy_loss"] + args.value_loss_weight * losses["value_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        step += 1
        samples += args.batch_size
        for key, value in losses.items():
            window[key] = window.get(key, 0.0) + value.detach()
        window_steps += 1
        if device.type == "cuda" and step % 50 == 0:
            torch.cuda.synchronize()
        train_time += time.perf_counter() - tick

        if train_time - last_val >= args.val_every_seconds:
            elapsed = time.perf_counter() - window_start
            row = {k: float(v) / window_steps for k, v in window.items()}
            row.update(
                validate(model, ds),
                lr=optimizer.param_groups[0]["lr"],
                samples_per_s=window_steps * args.batch_size / elapsed,
            )
            if train_time - last_game_eval >= args.game_eval_every_minutes * 60:
                row.update(game_eval())
                last_game_eval = train_time
            record(row)
            last_val, window, window_steps, window_start = (
                train_time,
                {},
                0,
                time.perf_counter(),
            )

    final = {**validate(model, ds), **game_eval(), "final": True}
    record(final)
    torch.save(
        {"model_state_dict": model.state_dict(), "args": asdict(args)},
        out_dir / "model.pt",
    )
    (out_dir / "final.json").write_text(json.dumps(final, indent=1))
    logger.info("Saved", path=str(out_dir / "model.pt"))
    if wandb_run is not None:
        wandb_run.finish()


def init_wandb(args: ScriptArgs, run_name: str, out_dir: Path):
    if not args.wandb_project:
        return None
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=asdict(args),
        dir=str(out_dir),
    )


def wandb_row(row: dict) -> dict:
    """Group metrics so W&B shows game attack, val and train curves as separate panels."""
    out = {}
    for key, value in row.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if key.startswith("attack") or key.startswith("game_"):
            out[f"game/{key}"] = value
        elif key.startswith("val_"):
            out[f"val/{key[4:]}"] = value
        else:
            out[f"train/{key}"] = value
    return out


def run_backfill(args: ScriptArgs) -> None:
    """Upload an existing run's metrics.jsonl to W&B (for runs started without logging)."""
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(args.checkpoint, map_location=device)
        model = BigPolicyNet(
            **{
                k: saved["args"][k]
                for k in ("channels", "blocks", "cond_dim", "head_channels")
            }
        )
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
