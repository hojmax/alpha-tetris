from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

import tetris_core.tetris_core as tetris_core
from tetris_bot.constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DEFAULT_CONFIG_PATH,
    NUM_ACTIONS,
    NUM_PIECE_TYPES,
    QUEUE_SIZE,
)
from tetris_bot.ml.config import load_training_config
from tetris_bot.ml.network import (
    AUX_FEATURES,
    COMBO_NORMALIZATION_MAX,
    PIECE_AUX_FEATURES,
    ConvBackbone,
    HeadsModel,
    TetrisNet,
    build_aux_features,
)
from tetris_bot.ml.weights import export_onnx, export_split_models


OVERHANG_NORMALIZATION_DIVISOR = 25.0
BUMPINESS_NORMALIZATION_DIVISOR = 200.0
HOLES_NORMALIZATION_DIVISOR = 20.0
TOTAL_BLOCKS_NORMALIZATION_DIVISOR = 60.0
COLUMN_HEIGHT_NORMALIZATION_DIVISOR = 8.0
MAX_COLUMN_HEIGHT_NORMALIZATION_DIVISOR = 20.0
ROW_FILL_FEATURE_ROWS = 4
ROW_FILL_FEATURE_START = BOARD_HEIGHT - ROW_FILL_FEATURE_ROWS


def _binary_board(env: tetris_core.TetrisEnv) -> np.ndarray:
    board = np.asarray(env.get_board(), dtype=np.float32)
    return np.where(board != 0.0, 1.0, 0.0).astype(np.float32)


def _hidden_piece_distribution(env: tetris_core.TetrisEnv) -> np.ndarray:
    visible_state = env.clone_state()
    visible_state.truncate_queue(QUEUE_SIZE)
    hidden_candidates = visible_state.get_possible_next_pieces()
    if len(hidden_candidates) == 0:
        raise ValueError("Hidden-piece candidate set must not be empty")

    hidden_distribution = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
    probability = 1.0 / float(len(hidden_candidates))
    for piece_type in hidden_candidates:
        hidden_distribution[int(piece_type)] = probability
    return hidden_distribution


def _compute_diagnostics(
    board: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, float, float, float, float]:
    height, width = board.shape
    filled = board != 0.0

    raw_column_heights = np.zeros(width, dtype=np.int32)
    for x in range(width):
        filled_rows = np.flatnonzero(filled[:, x])
        if filled_rows.size > 0:
            raw_column_heights[x] = height - int(filled_rows[0])

    raw_row_fill_counts = filled.sum(axis=1, dtype=np.int32)
    raw_total_blocks = int(raw_row_fill_counts.sum())

    raw_bumpiness = 0
    if width >= 2:
        deltas = np.diff(raw_column_heights)
        raw_bumpiness = int(np.sum(deltas * deltas))

    empty = ~filled
    reachable = np.zeros((height, width), dtype=bool)
    frontier: deque[tuple[int, int]] = deque()
    for x in range(width):
        if not empty[0, x]:
            continue
        reachable[0, x] = True
        frontier.append((0, x))

    while frontier:
        y, x = frontier.popleft()
        if y > 0 and empty[y - 1, x] and not reachable[y - 1, x]:
            reachable[y - 1, x] = True
            frontier.append((y - 1, x))
        if y + 1 < height and empty[y + 1, x] and not reachable[y + 1, x]:
            reachable[y + 1, x] = True
            frontier.append((y + 1, x))
        if x > 0 and empty[y, x - 1] and not reachable[y, x - 1]:
            reachable[y, x - 1] = True
            frontier.append((y, x - 1))
        if x + 1 < width and empty[y, x + 1] and not reachable[y, x + 1]:
            reachable[y, x + 1] = True
            frontier.append((y, x + 1))

    raw_overhang_fields = 0
    raw_holes = 0
    for x in range(width):
        seen_filled = False
        for y in range(height):
            if filled[y, x]:
                seen_filled = True
                continue
            if not seen_filled:
                continue
            raw_overhang_fields += 1
            if not reachable[y, x]:
                raw_holes += 1

    normalized_column_heights = (
        raw_column_heights.astype(np.float32) / COLUMN_HEIGHT_NORMALIZATION_DIVISOR
    )
    raw_max = int(np.max(raw_column_heights))
    max_column_height = float(raw_max) / MAX_COLUMN_HEIGHT_NORMALIZATION_DIVISOR
    normalized_row_fill_counts = raw_row_fill_counts[ROW_FILL_FEATURE_START:].astype(
        np.float32
    ) / float(width)
    normalized_total_blocks = (
        float(raw_total_blocks) / TOTAL_BLOCKS_NORMALIZATION_DIVISOR
    )
    normalized_bumpiness = float(raw_bumpiness) / BUMPINESS_NORMALIZATION_DIVISOR
    normalized_holes = float(raw_holes) / HOLES_NORMALIZATION_DIVISOR
    normalized_overhang_fields = (
        float(raw_overhang_fields) / OVERHANG_NORMALIZATION_DIVISOR
    )
    return (
        normalized_column_heights,
        max_column_height,
        normalized_row_fill_counts,
        normalized_total_blocks,
        normalized_bumpiness,
        normalized_holes,
        normalized_overhang_fields,
    )


def _encode_state_python(
    env: tetris_core.TetrisEnv,
    move_number: int,
    max_placements: int,
) -> tuple[np.ndarray, np.ndarray]:
    board_matrix = _binary_board(env)
    board = board_matrix.reshape(-1)

    current = env.get_current_piece()
    hold = env.get_hold_piece()
    queue = env.get_queue(QUEUE_SIZE)

    current_piece = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
    current_piece_idx = 0 if current is None else current.piece_type
    current_piece[current_piece_idx] = 1.0

    hold_piece = np.zeros(NUM_PIECE_TYPES + 1, dtype=np.float32)
    if hold is None:
        hold_piece[NUM_PIECE_TYPES] = 1.0
    else:
        hold_piece[hold.piece_type] = 1.0

    queue_features = np.zeros((QUEUE_SIZE, NUM_PIECE_TYPES), dtype=np.float32)
    for slot, piece_type in enumerate(queue):
        queue_features[slot, piece_type] = 1.0

    (
        column_heights,
        max_column_height,
        row_fill_counts,
        total_blocks,
        bumpiness,
        holes,
        overhang_fields,
    ) = _compute_diagnostics(board_matrix)

    aux = build_aux_features(
        current_piece=current_piece,
        hold_piece=hold_piece,
        hold_available=1.0 if not env.is_hold_used() else 0.0,
        next_queue=queue_features,
        placement_count=float(move_number) / float(max_placements),
        combo_feature=float(env.combo) / COMBO_NORMALIZATION_MAX,
        back_to_back=1.0 if env.back_to_back else 0.0,
        next_hidden_piece_probs=_hidden_piece_distribution(env),
        column_heights=column_heights,
        max_column_height=max_column_height,
        row_fill_counts=row_fill_counts,
        total_blocks=total_blocks,
        bumpiness=bumpiness,
        holes=holes,
        overhang_fields=overhang_fields,
    )

    if aux.size != AUX_FEATURES:
        raise ValueError(f"Expected {AUX_FEATURES} aux features, got {aux.size}")
    return board.astype(np.float32), aux.astype(np.float32)


def _python_masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_logits = np.where(mask, logits, -np.inf).astype(np.float32)
    probs = torch.softmax(torch.from_numpy(masked_logits), dim=0)
    return probs.numpy()


def _build_env_variants() -> list[tuple[tetris_core.TetrisEnv, int, int]]:
    env_a = tetris_core.TetrisEnv.with_seed(10, 20, 1)
    env_b = tetris_core.TetrisEnv.with_seed(10, 20, 2)
    env_b.hold()
    env_c = tetris_core.TetrisEnv.with_seed(10, 20, 3)
    env_c.rotate_cw()
    env_c.move_left()
    env_c.hard_drop()
    env_d = tetris_core.TetrisEnv.with_seed(10, 20, 4)
    env_d.rotate_ccw()
    env_d.move_right()
    env_d.hard_drop()
    env_d.hold()
    env_d.rotate_cw()

    return [
        (env_a, 0, 100),
        (env_b, 1, 100),
        (env_c, 37, 100),
        (env_d, 99, 100),
    ]


def test_encode_state_matches_between_rust_inference_and_python_training_view() -> None:
    for env, move_number, max_placements in _build_env_variants():
        env.placement_count = move_number
        rust_board, rust_aux = tetris_core.debug_encode_state(env, max_placements)
        py_board, py_aux = _encode_state_python(env, move_number, max_placements)

        np.testing.assert_array_equal(
            np.asarray(rust_board, dtype=np.float32), py_board
        )
        np.testing.assert_array_equal(np.asarray(rust_aux, dtype=np.float32), py_aux)


def test_masked_softmax_matches_between_rust_and_python() -> None:
    rng = np.random.default_rng(123)
    logits = rng.normal(size=NUM_ACTIONS).astype(np.float32)
    mask = rng.random(NUM_ACTIONS) > 0.2
    mask[0] = True

    rust_probs = np.asarray(
        tetris_core.debug_masked_softmax(logits.tolist(), mask.tolist()),
        dtype=np.float32,
    )
    py_probs = _python_masked_softmax(logits, mask)

    np.testing.assert_allclose(rust_probs, py_probs, rtol=1e-6, atol=1e-7)
    assert np.all(rust_probs[~mask] == 0.0)
    np.testing.assert_allclose(np.sum(rust_probs), 1.0, rtol=1e-6, atol=1e-7)


def test_pytorch_and_rust_tract_inference_match_on_same_onnx(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = TetrisNet(
        trunk_channels=16,
        num_conv_residual_blocks=1,
        reduction_channels=32,
        fc_hidden=128,
        conv_kernel_size=3,
        conv_padding=1,
    )
    model.eval()

    onnx_path = tmp_path / "parity_model.onnx"
    assert export_onnx(model, onnx_path)
    assert export_split_models(model, onnx_path)

    for env, move_number, max_placements in _build_env_variants():
        board, aux = _encode_state_python(env, move_number, max_placements)
        action_mask = np.asarray(tetris_core.debug_get_action_mask(env), dtype=bool)

        board_tensor = torch.from_numpy(board).reshape(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
        aux_tensor = torch.from_numpy(aux).reshape(1, -1)
        mask_tensor = torch.from_numpy(action_mask).reshape(1, -1)

        with torch.no_grad():
            policy_logits, value_tensor = model(board_tensor, aux_tensor)
            masked_logits = policy_logits.masked_fill(~mask_tensor, float("-inf"))
            expected_policy = torch.softmax(masked_logits, dim=-1).squeeze(0).numpy()
            expected_value = value_tensor.item()

        rust_policy, rust_value = tetris_core.debug_predict_masked_from_tensors(
            str(onnx_path),
            board.tolist(),
            aux.tolist(),
            action_mask.tolist(),
        )

        np.testing.assert_allclose(
            np.asarray(rust_policy, dtype=np.float32),
            expected_policy.astype(np.float32),
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            float(rust_value), float(expected_value), rtol=1e-4, atol=2.5e-5
        )


def test_split_model_matches_end_to_end_pytorch(tmp_path: Path) -> None:
    """Verify split computation (conv + board proj + heads) matches TetrisNet.forward()."""
    rng = np.random.default_rng(42)

    for seed in range(10):
        torch.manual_seed(seed)
        model = TetrisNet(
            trunk_channels=16,
            num_conv_residual_blocks=1,
            reduction_channels=32,
            fc_hidden=128,
            conv_kernel_size=3,
            conv_padding=1,
        )
        model.eval()

        conv_backbone = ConvBackbone(model)
        conv_backbone.eval()
        heads = HeadsModel(model)
        heads.eval()

        # Generate random inputs
        for _ in range(20):
            board = torch.from_numpy(
                rng.integers(0, 2, size=(1, 1, BOARD_HEIGHT, BOARD_WIDTH)).astype(
                    np.float32
                )
            )
            aux = torch.from_numpy(
                rng.standard_normal((1, AUX_FEATURES)).astype(np.float32)
            )

            with torch.no_grad():
                # End-to-end
                expected_logits, expected_value = model(board, aux)

                # Split path
                conv_out = conv_backbone(board)  # (1, 1600)
                board_stats = aux[:, PIECE_AUX_FEATURES:]  # (1, 19)
                board_h = model.forward_board_embedding_from_parts(
                    conv_out, board_stats
                )
                piece_aux = aux[:, :PIECE_AUX_FEATURES]  # (1, 61)
                split_logits, split_value = heads(board_h, piece_aux)

            np.testing.assert_allclose(
                split_logits.numpy(),
                expected_logits.numpy(),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Policy logits mismatch (seed={seed})",
            )
            np.testing.assert_allclose(
                split_value.numpy(),
                expected_value.numpy(),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Value mismatch (seed={seed})",
            )


_SMALL_NETWORK_KWARGS = {
    "trunk_channels": 16,
    "num_conv_residual_blocks": 1,
    "reduction_channels": 32,
    "fc_hidden": 128,
    "conv_kernel_size": 3,
    "conv_padding": 1,
}


@pytest.mark.parametrize(
    "network_kwargs",
    [
        pytest.param(_SMALL_NETWORK_KWARGS, id="small"),
        pytest.param(
            load_training_config(DEFAULT_CONFIG_PATH).network.model_dump(),
            id="default_config",
        ),
    ],
)
def test_split_onnx_rust_matches_end_to_end_pytorch(
    tmp_path: Path, network_kwargs: dict
) -> None:
    """Verify Rust split inference matches PyTorch end-to-end on real game states."""
    torch.manual_seed(99)
    model = TetrisNet(**network_kwargs)
    with torch.no_grad():
        # Non-trivial GroupNorm/LayerNorm affines so the native kernels see real scales.
        for name, param in model.named_parameters():
            if "bn" in name or "ln" in name:
                param.add_(0.2 * torch.randn_like(param))
    model.eval()

    onnx_path = tmp_path / "split_parity.onnx"
    assert export_onnx(model, onnx_path)
    assert export_split_models(model, onnx_path)

    # Build diverse game states
    envs: list[tuple[tetris_core.TetrisEnv, int, int]] = _build_env_variants()
    # Add more states with different seeds and more drops
    for game_seed in range(10, 30):
        env = tetris_core.TetrisEnv.with_seed(10, 20, game_seed)
        for _ in range(game_seed % 5):
            env.hard_drop()
        if game_seed % 3 == 0:
            env.hold()
        envs.append((env, game_seed % 100, 100))

    for env, move_number, max_placements in envs:
        board, aux = _encode_state_python(env, move_number, max_placements)
        action_mask = np.asarray(tetris_core.debug_get_action_mask(env), dtype=bool)
        if not action_mask.any():
            continue

        # PyTorch end-to-end
        board_tensor = torch.from_numpy(board).reshape(1, 1, BOARD_HEIGHT, BOARD_WIDTH)
        aux_tensor = torch.from_numpy(aux).reshape(1, -1)
        mask_tensor = torch.from_numpy(action_mask).reshape(1, -1)

        with torch.no_grad():
            policy_logits, value_tensor = model(board_tensor, aux_tensor)
            masked_logits = policy_logits.masked_fill(~mask_tensor, float("-inf"))
            expected_policy = torch.softmax(masked_logits, dim=-1).squeeze(0).numpy()
            expected_value = value_tensor.item()

        # Rust split inference
        rust_policy, rust_value = tetris_core.debug_predict_masked_from_tensors(
            str(onnx_path),
            board.tolist(),
            aux.tolist(),
            action_mask.tolist(),
        )

        np.testing.assert_allclose(
            np.asarray(rust_policy, dtype=np.float32),
            expected_policy.astype(np.float32),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Policy mismatch (seed={move_number})",
        )
        np.testing.assert_allclose(
            float(rust_value),
            float(expected_value),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"Value mismatch (seed={move_number})",
        )


def _piece_state(piece: object | None) -> tuple[int, int, int, int] | None:
    if piece is None:
        return None
    return (piece.piece_type, piece.x, piece.y, piece.rotation)


def _assert_env_states_equal(
    lhs: tetris_core.TetrisEnv, rhs: tetris_core.TetrisEnv
) -> None:
    assert lhs.width == rhs.width
    assert lhs.height == rhs.height
    assert lhs.lines_cleared == rhs.lines_cleared
    assert lhs.game_over == rhs.game_over
    assert lhs.attack == rhs.attack
    assert lhs.combo == rhs.combo
    assert lhs.back_to_back == rhs.back_to_back

    assert lhs.get_board() == rhs.get_board()
    assert lhs.get_board_piece_types() == rhs.get_board_piece_types()
    assert _piece_state(lhs.get_current_piece()) == _piece_state(
        rhs.get_current_piece()
    )
    assert _piece_state(lhs.get_hold_piece()) == _piece_state(rhs.get_hold_piece())
    assert lhs.is_hold_used() == rhs.is_hold_used()
    assert lhs.get_pieces_spawned() == rhs.get_pieces_spawned()
    assert lhs.get_queue_len() == rhs.get_queue_len()
    assert lhs.get_queue(lhs.get_queue_len()) == rhs.get_queue(rhs.get_queue_len())
    assert lhs.get_possible_next_pieces() == rhs.get_possible_next_pieces()


def _assert_tree_exports_equal(
    lhs: tetris_core.MCTSTreeExport, rhs: tetris_core.MCTSTreeExport
) -> None:
    assert lhs.root_id == rhs.root_id
    assert lhs.num_simulations == rhs.num_simulations
    assert lhs.selected_action == rhs.selected_action
    np.testing.assert_array_equal(
        np.asarray(lhs.policy, dtype=np.float32),
        np.asarray(rhs.policy, dtype=np.float32),
    )

    assert len(lhs.nodes) == len(rhs.nodes)
    for lhs_node, rhs_node in zip(lhs.nodes, rhs.nodes, strict=True):
        assert lhs_node.id == rhs_node.id
        assert lhs_node.node_type == rhs_node.node_type
        assert lhs_node.visit_count == rhs_node.visit_count
        assert lhs_node.value_sum == rhs_node.value_sum
        assert lhs_node.mean_value == rhs_node.mean_value
        np.testing.assert_array_equal(
            np.asarray(lhs_node.value_history, dtype=np.float32),
            np.asarray(rhs_node.value_history, dtype=np.float32),
        )
        assert lhs_node.nn_value == rhs_node.nn_value
        assert lhs_node.is_terminal == rhs_node.is_terminal
        assert lhs_node.move_number == rhs_node.move_number
        assert lhs_node.attack == rhs_node.attack
        assert lhs_node.parent_id == rhs_node.parent_id
        assert lhs_node.edge_from_parent == rhs_node.edge_from_parent
        assert lhs_node.children == rhs_node.children
        assert lhs_node.valid_actions == rhs_node.valid_actions
        np.testing.assert_array_equal(
            np.asarray(lhs_node.action_priors, dtype=np.float32),
            np.asarray(rhs_node.action_priors, dtype=np.float32),
        )
        _assert_env_states_equal(lhs_node.state, rhs_node.state)


def test_mcts_tree_cache_parity_matches_uncached_search(tmp_path: Path) -> None:
    torch.manual_seed(2026)
    model = TetrisNet(
        trunk_channels=16,
        num_conv_residual_blocks=1,
        reduction_channels=32,
        fc_hidden=128,
        conv_kernel_size=3,
        conv_padding=1,
    )
    model.eval()

    onnx_path = tmp_path / "mcts_cache_parity.onnx"
    assert export_onnx(model, onnx_path)
    assert export_split_models(model, onnx_path)

    env = tetris_core.TetrisEnv.with_seed(10, 20, 4242)
    for _ in range(5):
        env.rotate_cw()
        env.move_left()
        env.hard_drop()
    env.hold()
    env.rotate_ccw()
    env.move_right()

    config = tetris_core.MCTSConfig()
    config.num_simulations = 256
    config.c_puct = 1.5
    config.temperature = 1.0
    config.dirichlet_alpha = 0.15
    config.dirichlet_epsilon = 0.25
    config.seed = 999
    config.max_placements = 100
    config.track_value_history = True
    config.death_penalty = 0.0

    cached_agent = tetris_core.MCTSAgent(config)
    assert cached_agent.load_model(str(onnx_path))
    assert cached_agent.set_board_cache_enabled(True)

    warmup_env = env.clone_state()
    warmup_env.placement_count = 17
    warmup = cached_agent.search_with_tree(warmup_env, add_noise=False)
    assert warmup is not None
    warmup_stats = cached_agent.get_and_reset_cache_stats()
    assert warmup_stats is not None

    cached_env = env.clone_state()
    cached_env.placement_count = 17
    cached_search = cached_agent.search_with_tree(cached_env, add_noise=False)
    assert cached_search is not None
    cached_result, cached_tree = cached_search
    cached_stats = cached_agent.get_and_reset_cache_stats()
    assert cached_stats is not None
    cache_hits, cache_misses, cache_size = cached_stats
    assert cache_hits > 0
    assert cache_misses == 0
    assert cache_size > 0

    uncached_agent = tetris_core.MCTSAgent(config)
    assert uncached_agent.load_model(str(onnx_path))
    assert uncached_agent.set_board_cache_enabled(False)
    uncached_env = env.clone_state()
    uncached_env.placement_count = 17
    uncached_search = uncached_agent.search_with_tree(uncached_env, add_noise=False)
    assert uncached_search is not None
    uncached_result, uncached_tree = uncached_search
    uncached_stats = uncached_agent.get_and_reset_cache_stats()
    assert uncached_stats is not None
    uncached_hits, uncached_misses, uncached_size = uncached_stats
    assert uncached_hits == 0
    assert uncached_misses == 0
    assert uncached_size == 0

    assert cached_result.action == uncached_result.action
    assert cached_result.value == uncached_result.value
    assert cached_result.num_simulations == uncached_result.num_simulations
    np.testing.assert_array_equal(
        np.asarray(cached_result.policy, dtype=np.float32),
        np.asarray(uncached_result.policy, dtype=np.float32),
    )

    _assert_tree_exports_equal(cached_tree, uncached_tree)


def test_direct_inference_cache_matches_uncached(tmp_path: Path) -> None:
    model = TetrisNet(
        trunk_channels=8,
        num_conv_residual_blocks=2,
        reduction_channels=16,
        fc_hidden=128,
        aux_hidden=64,
        num_fusion_blocks=1,
        conv_kernel_size=3,
        conv_padding=1,
    )
    model.eval()

    onnx_path = tmp_path / "direct_inference_cache_parity.onnx"
    assert export_onnx(model, onnx_path)
    assert export_split_models(model, onnx_path)

    env = tetris_core.TetrisEnv.with_seed(10, 20, 20260318)
    for step_idx in range(6):
        action_mask = tetris_core.debug_get_action_mask(env)
        valid_actions = [idx for idx, is_valid in enumerate(action_mask) if is_valid]
        assert valid_actions, (
            "Expected at least one valid action while building test state"
        )
        chosen_action = valid_actions[min(step_idx, len(valid_actions) - 1)]
        assert env.execute_action_index(chosen_action) is not None

    config = tetris_core.MCTSConfig()
    config.max_placements = 100

    cached_agent = tetris_core.MCTSAgent(config)
    assert cached_agent.load_model(str(onnx_path))
    assert cached_agent.set_board_cache_enabled(True)

    cached_policy_first, cached_value_first = cached_agent.predict_with_valid_actions(
        env.clone_state(), max_placements=100
    )
    first_stats = cached_agent.get_and_reset_cache_stats()
    assert first_stats is not None
    first_hits, first_misses, first_cache_size = first_stats
    assert first_hits == 0
    assert first_misses == 1
    assert first_cache_size > 0

    cached_policy_second, cached_value_second = cached_agent.predict_with_valid_actions(
        env.clone_state(), max_placements=100
    )
    second_stats = cached_agent.get_and_reset_cache_stats()
    assert second_stats is not None
    second_hits, second_misses, second_cache_size = second_stats
    assert second_hits == 1
    assert second_misses == 0
    assert second_cache_size > 0

    uncached_agent = tetris_core.MCTSAgent(config)
    assert uncached_agent.load_model(str(onnx_path))
    assert uncached_agent.set_board_cache_enabled(False)
    uncached_policy, uncached_value = uncached_agent.predict_with_valid_actions(
        env.clone_state(), max_placements=100
    )
    uncached_stats = uncached_agent.get_and_reset_cache_stats()
    assert uncached_stats is not None
    uncached_hits, uncached_misses, uncached_cache_size = uncached_stats
    assert uncached_hits == 0
    assert uncached_misses == 0
    assert uncached_cache_size == 0

    np.testing.assert_allclose(
        np.asarray(cached_policy_first, dtype=np.float32),
        np.asarray(cached_policy_second, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(cached_policy_first, dtype=np.float32),
        np.asarray(uncached_policy, dtype=np.float32),
    )
    assert cached_value_first == cached_value_second == uncached_value
