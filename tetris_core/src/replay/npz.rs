//! NPZ File I/O
//!
//! Read and write replay examples in NPZ format (compatible with Python numpy).
//!
//! The policy and legal-action mask are stored in the compact form the replay buffer
//! keeps in memory (see [`crate::replay::compact`]): `action_mask_bits` (N, 84) uint8 and
//! `policy_legal` (N, 64) uint16. Snapshots written before that change carry dense
//! `policy_targets` / `action_masks` (672 wide, or the legacy 734 wide); the reader converts
//! those on the fly so existing runs can still be resumed and warm-started from.
//!
//! Reading streams every column straight into [`CompactExample`]s, so peak memory is the
//! compact buffer itself rather than the decoded dense arrays.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use npyz::{NpyFile, WriterBuilder};
use zip::read::{ZipArchive, ZipFile};
use zip::write::{FileOptions, ZipWriter};
use zip::CompressionMethod;

use crate::game::action_space::{
    get_action_space, HOLD_ACTION_INDEX, LEGACY_HOLD_ACTION_INDEX, LEGACY_NUM_ACTIONS,
    LEGACY_NUM_PLACEMENT_ACTIONS,
};
use crate::game::constants::{
    BOARD_HEIGHT, BOARD_WIDTH, NUM_PIECE_TYPES, QUEUE_SIZE, ROW_FILL_FEATURE_ROWS,
};
use crate::replay::compact::{
    get_bit, pack_board, pack_policy, CompactExample, BOARD_BYTES, BOARD_CELLS, EMPTY_QUEUE_SLOT,
    MASK_BYTES, POLICY_SLOTS,
};
use crate::search::NUM_ACTIONS;

type Archive = ZipArchive<BufReader<File>>;

/// Write replay examples to NPZ format (compatible with Python numpy).
pub fn write_examples_to_npz(filepath: &Path, examples: &[CompactExample]) -> Result<(), String> {
    write_examples_slices_to_npz(filepath, examples, &[])
}

/// Streaming NPZ writer that accepts two contiguous slices (to support VecDeque::as_slices()).
///
/// Each array is streamed directly from the examples to the zip file — no intermediate
/// column vectors are allocated.
///
/// Layout:
/// - boards: (N, 20, 10) uint8
/// - current_pieces: (N, 7) float32 one-hot
/// - hold_pieces: (N, 8) float32 one-hot
/// - hold_available: (N,) uint8
/// - next_queue: (N, 5, 7) float32 one-hot (all-zero rows for missing slots)
/// - move_numbers: (N,) uint32 raw frame indices (includes holds)
/// - placement_counts: (N,) float32 normalized (placement_idx / max_placements, [0,1])
/// - combos: (N,) float32 normalized (combo / COMBO_NORMALIZATION_MAX, uncapped)
/// - back_to_back: (N,) uint8
/// - next_hidden_piece_probs: (N, 7) float32
/// - column_heights: (N, 10) float32 normalized by COLUMN_HEIGHT_NORMALIZATION_DIVISOR (uncapped)
/// - max_column_heights: (N,) float32
/// - row_fill_counts: (N, 4) float32 normalized by board width (bottom rows only)
/// - total_blocks: (N,) float32 normalized by TOTAL_BLOCKS_NORMALIZATION_DIVISOR (uncapped)
/// - bumpiness: (N,) float32 normalized by BUMPINESS_NORMALIZATION_DIVISOR (uncapped)
/// - holes: (N,) float32 normalized by HOLES_NORMALIZATION_DIVISOR (uncapped)
/// - action_mask_bits: (N, 84) uint8, legal actions bit-packed with little bit order
/// - policy_legal: (N, 64) uint16, policy of the i-th legal action (ascending action
///   index) as `round(p * 65535)`; unused slots are 0
/// - value_targets: (N,) float32
/// - overhang_fields: (N,) float32 normalized by OVERHANG_NORMALIZATION_DIVISOR (uncapped)
/// - game_numbers: (N,) uint64 (1-indexed game IDs aligned with WandB game_number)
/// - game_total_attacks: (N,) uint32 (raw total attack for each example's source game)
pub(crate) fn write_examples_slices_to_npz(
    filepath: &Path,
    examples_a: &[CompactExample],
    examples_b: &[CompactExample],
) -> Result<(), String> {
    let n = examples_a.len() + examples_b.len();
    if n == 0 {
        return Ok(());
    }

    let examples = || examples_a.iter().chain(examples_b.iter());
    let n64 = n as u64;

    let file = File::create(filepath).map_err(|e| e.to_string())?;
    let mut zip = ZipWriter::new(BufWriter::new(file));
    // Snapshots of large buffers can exceed 4 GiB, so ZIP64 must be enabled.
    // Deflate level 1: higher levels burn far more CPU for little extra compression.
    let options = FileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .compression_level(Some(1))
        .large_file(true);
    let z = &mut zip;

    write_flat_field(
        z,
        options,
        "boards.npy",
        n64,
        &[BOARD_HEIGHT as u64, BOARD_WIDTH as u64],
        examples()
            .flat_map(|ex| (0..BOARD_CELLS).map(|cell| u8::from(get_bit(&ex.board_bits, cell)))),
    )?;
    write_flat_field(
        z,
        options,
        "current_pieces.npy",
        n64,
        &[NUM_PIECE_TYPES as u64],
        examples().flat_map(|ex| one_hot::<NUM_PIECE_TYPES>(Some(ex.current_piece.into()))),
    )?;
    write_flat_field(
        z,
        options,
        "hold_pieces.npy",
        n64,
        &[(NUM_PIECE_TYPES + 1) as u64],
        examples().flat_map(|ex| one_hot::<{ NUM_PIECE_TYPES + 1 }>(Some(ex.hold_piece.into()))),
    )?;
    write_scalar_field(
        z,
        options,
        "hold_available.npy",
        n64,
        examples().map(|ex| ex.hold_available as u8),
    )?;
    write_flat_field(
        z,
        options,
        "next_queue.npy",
        n64,
        &[QUEUE_SIZE as u64, NUM_PIECE_TYPES as u64],
        examples().flat_map(|ex| {
            ex.next_queue.into_iter().flat_map(|piece| {
                one_hot::<NUM_PIECE_TYPES>((piece != EMPTY_QUEUE_SLOT).then_some(piece.into()))
            })
        }),
    )?;
    write_scalar_field(
        z,
        options,
        "move_numbers.npy",
        n64,
        examples().map(|ex| ex.move_number),
    )?;
    write_scalar_field(
        z,
        options,
        "placement_counts.npy",
        n64,
        examples().map(|ex| ex.placement_count),
    )?;
    write_scalar_field(z, options, "combos.npy", n64, examples().map(|ex| ex.combo))?;
    write_scalar_field(
        z,
        options,
        "back_to_back.npy",
        n64,
        examples().map(|ex| ex.back_to_back as u8),
    )?;
    write_flat_field(
        z,
        options,
        "next_hidden_piece_probs.npy",
        n64,
        &[NUM_PIECE_TYPES as u64],
        examples().flat_map(|ex| ex.next_hidden_piece_probs),
    )?;
    write_flat_field(
        z,
        options,
        "column_heights.npy",
        n64,
        &[BOARD_WIDTH as u64],
        examples().flat_map(|ex| ex.column_heights),
    )?;
    write_scalar_field(
        z,
        options,
        "max_column_heights.npy",
        n64,
        examples().map(|ex| ex.max_column_height),
    )?;
    write_flat_field(
        z,
        options,
        "row_fill_counts.npy",
        n64,
        &[ROW_FILL_FEATURE_ROWS as u64],
        examples().flat_map(|ex| ex.row_fill_counts),
    )?;
    write_scalar_field(
        z,
        options,
        "total_blocks.npy",
        n64,
        examples().map(|ex| ex.total_blocks),
    )?;
    write_scalar_field(
        z,
        options,
        "bumpiness.npy",
        n64,
        examples().map(|ex| ex.bumpiness),
    )?;
    write_scalar_field(z, options, "holes.npy", n64, examples().map(|ex| ex.holes))?;
    write_flat_field(
        z,
        options,
        "action_mask_bits.npy",
        n64,
        &[MASK_BYTES as u64],
        examples().flat_map(|ex| ex.mask_bits),
    )?;
    write_flat_field(
        z,
        options,
        "policy_legal.npy",
        n64,
        &[POLICY_SLOTS as u64],
        examples().flat_map(|ex| ex.policy_legal),
    )?;
    write_scalar_field(
        z,
        options,
        "value_targets.npy",
        n64,
        examples().map(|ex| ex.value),
    )?;
    write_scalar_field(
        z,
        options,
        "overhang_fields.npy",
        n64,
        examples().map(|ex| ex.overhang_fields),
    )?;
    write_scalar_field(
        z,
        options,
        "game_numbers.npy",
        n64,
        examples().map(|ex| ex.game_number),
    )?;
    write_scalar_field(
        z,
        options,
        "game_total_attacks.npy",
        n64,
        examples().map(|ex| ex.game_total_attack),
    )?;

    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

fn blank_example() -> CompactExample {
    CompactExample {
        board_bits: [0; BOARD_BYTES],
        current_piece: 0,
        hold_piece: NUM_PIECE_TYPES as u8,
        hold_available: false,
        back_to_back: false,
        next_queue: [EMPTY_QUEUE_SLOT; QUEUE_SIZE],
        move_number: 0,
        placement_count: 0.0,
        combo: 0.0,
        next_hidden_piece_probs: [0.0; NUM_PIECE_TYPES],
        column_heights: [0.0; BOARD_WIDTH],
        max_column_height: 0.0,
        row_fill_counts: [0.0; ROW_FILL_FEATURE_ROWS],
        total_blocks: 0.0,
        bumpiness: 0.0,
        holes: 0.0,
        overhang_fields: 0.0,
        value: 0.0,
        game_number: 0,
        game_total_attack: 0,
        mask_bits: [0; MASK_BYTES],
        policy_legal: [0; POLICY_SLOTS],
    }
}

/// Read replay examples from NPZ format (compact, dense, or legacy-width dense policy).
pub fn read_examples_from_npz(filepath: &Path) -> Result<Vec<CompactExample>, String> {
    let file = File::open(filepath).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(BufReader::new(file)).map_err(|e| e.to_string())?;
    let is_compact = archive.file_names().any(|name| name == "policy_legal.npy");

    let shape = npy_shape(&mut archive, "boards")?;
    let n = validate_shape_with_dynamic_batch(
        "boards",
        &shape,
        &[0, BOARD_HEIGHT as u64, BOARD_WIDTH as u64],
    )?;
    let mut ex = vec![blank_example(); n];
    let a = &mut archive;

    let board_dims = [BOARD_HEIGHT as u64, BOARD_WIDTH as u64];
    for_each_row::<u8>(a, "boards", n, &board_dims, |i, row| {
        ex[i].board_bits = pack_board(row);
    })?;
    for_each_row::<f32>(
        a,
        "current_pieces",
        n,
        &[NUM_PIECE_TYPES as u64],
        |i, row| {
            ex[i].current_piece = argmax_index(row) as u8;
        },
    )?;
    let hold_dims = [(NUM_PIECE_TYPES + 1) as u64];
    for_each_row::<f32>(a, "hold_pieces", n, &hold_dims, |i, row| {
        ex[i].hold_piece = argmax_index(row).min(NUM_PIECE_TYPES) as u8;
    })?;
    for_each_bool_like(a, "hold_available", n, |i, value| {
        ex[i].hold_available = value;
    })?;
    let queue_dims = [QUEUE_SIZE as u64, NUM_PIECE_TYPES as u64];
    for_each_row::<f32>(a, "next_queue", n, &queue_dims, |i, row| {
        for (slot, piece) in row.chunks_exact(NUM_PIECE_TYPES).enumerate() {
            ex[i].next_queue[slot] = if piece.iter().any(|&v| v > 0.0) {
                argmax_index(piece) as u8
            } else {
                EMPTY_QUEUE_SLOT
            };
        }
    })?;
    for_each_row::<u32>(a, "move_numbers", n, &[], |i, row| {
        ex[i].move_number = row[0];
    })?;
    for_each_row::<f32>(a, "placement_counts", n, &[], |i, row| {
        ex[i].placement_count = row[0];
    })?;
    for_each_row::<f32>(a, "combos", n, &[], |i, row| ex[i].combo = row[0])?;
    for_each_bool_like(a, "back_to_back", n, |i, value| ex[i].back_to_back = value)?;
    let piece_dims = [NUM_PIECE_TYPES as u64];
    for_each_row::<f32>(a, "next_hidden_piece_probs", n, &piece_dims, |i, row| {
        ex[i].next_hidden_piece_probs.copy_from_slice(row);
    })?;
    for_each_row::<f32>(a, "column_heights", n, &[BOARD_WIDTH as u64], |i, row| {
        ex[i].column_heights.copy_from_slice(row);
    })?;
    for_each_row::<f32>(a, "max_column_heights", n, &[], |i, row| {
        ex[i].max_column_height = row[0];
    })?;
    let fill_dims = [ROW_FILL_FEATURE_ROWS as u64];
    for_each_row::<f32>(a, "row_fill_counts", n, &fill_dims, |i, row| {
        ex[i].row_fill_counts.copy_from_slice(row);
    })?;
    for_each_row::<f32>(a, "total_blocks", n, &[], |i, row| {
        ex[i].total_blocks = row[0];
    })?;
    for_each_row::<f32>(a, "bumpiness", n, &[], |i, row| ex[i].bumpiness = row[0])?;
    for_each_row::<f32>(a, "holes", n, &[], |i, row| ex[i].holes = row[0])?;
    for_each_row::<f32>(a, "value_targets", n, &[], |i, row| ex[i].value = row[0])?;
    for_each_row::<f32>(a, "overhang_fields", n, &[], |i, row| {
        ex[i].overhang_fields = row[0];
    })?;
    for_each_row::<u64>(a, "game_numbers", n, &[], |i, row| {
        ex[i].game_number = row[0];
    })?;
    for_each_row::<u32>(a, "game_total_attacks", n, &[], |i, row| {
        ex[i].game_total_attack = row[0];
    })?;

    if is_compact {
        for_each_row::<u8>(a, "action_mask_bits", n, &[MASK_BYTES as u64], |i, row| {
            ex[i].mask_bits.copy_from_slice(row);
        })?;
        for_each_row::<u16>(a, "policy_legal", n, &[POLICY_SLOTS as u64], |i, row| {
            ex[i].policy_legal.copy_from_slice(row);
        })?;
    } else {
        read_dense_policy_into(a, &mut ex)?;
    }

    for (i, example) in ex.iter().enumerate() {
        if !example.overhang_fields.is_finite() || example.overhang_fields < 0.0 {
            return Err(format!(
                "overhang_fields[{}] must be finite and >= 0, got {}",
                i, example.overhang_fields
            ));
        }
    }
    Ok(ex)
}

/// Convert dense `policy_targets` / `action_masks` (current or legacy width) row by row.
fn read_dense_policy_into(archive: &mut Archive, ex: &mut [CompactExample]) -> Result<(), String> {
    let n = ex.len();
    let shape = npy_shape(archive, "action_masks")?;
    if shape.len() != 2 {
        return Err(format!("action_masks has rank {}, expected 2", shape.len()));
    }
    let width = shape[1] as usize;
    if width != NUM_ACTIONS && width != LEGACY_NUM_ACTIONS {
        return Err(format!(
            "policy_targets/action_masks width must be {} (current) or {} (legacy), got {}",
            NUM_ACTIONS, LEGACY_NUM_ACTIONS, width
        ));
    }
    // Masks are buffered bit-packed so the policy pass can pair each row with its mask.
    let row_bytes = width.div_ceil(8);
    let mut mask_rows = vec![0u8; n * row_bytes];
    for_each_bool_like_row(archive, "action_masks", n, width, |i, row| {
        for (action, &value) in row.iter().enumerate() {
            if value != 0 {
                mask_rows[i * row_bytes + action / 8] |= 1 << (action % 8);
            }
        }
    })?;

    let action_space = get_action_space();
    let mut mask_row = vec![0u8; width];
    let mut failure: Option<String> = None;
    for_each_row::<f32>(archive, "policy_targets", n, &[width as u64], |i, row| {
        if failure.is_some() {
            return;
        }
        let bits = &mask_rows[i * row_bytes..(i + 1) * row_bytes];
        for (action, slot) in mask_row.iter_mut().enumerate() {
            *slot = u8::from(get_bit(bits, action));
        }
        let (policy, mask) = if width == NUM_ACTIONS {
            (row.to_vec(), mask_row.iter().map(|&m| m != 0).collect())
        } else {
            let current_piece = usize::from(ex[i].current_piece);
            match adapt_legacy_policy_and_mask(action_space, current_piece, row, &mask_row) {
                Ok(adapted) => adapted,
                Err(error) => {
                    failure = Some(error);
                    return;
                }
            }
        };
        let (mask_bits, policy_legal) = pack_policy(&policy, &mask);
        ex[i].mask_bits = mask_bits;
        ex[i].policy_legal = policy_legal;
    })?;
    failure.map_or(Ok(()), Err)
}

fn adapt_legacy_policy_and_mask(
    action_space: &crate::game::action_space::ActionSpace,
    current_piece: usize,
    legacy_policy: &[f32],
    legacy_action_mask: &[u8],
) -> Result<(Vec<f32>, Vec<bool>), String> {
    if legacy_policy.len() != LEGACY_NUM_ACTIONS {
        return Err(format!(
            "legacy policy width mismatch: expected {}, got {}",
            LEGACY_NUM_ACTIONS,
            legacy_policy.len()
        ));
    }
    if legacy_action_mask.len() != LEGACY_NUM_ACTIONS {
        return Err(format!(
            "legacy action mask width mismatch: expected {}, got {}",
            LEGACY_NUM_ACTIONS,
            legacy_action_mask.len()
        ));
    }

    let mut policy = vec![0.0; NUM_ACTIONS];
    let mut action_mask = vec![false; NUM_ACTIONS];
    policy[HOLD_ACTION_INDEX] = legacy_policy[LEGACY_HOLD_ACTION_INDEX];
    action_mask[HOLD_ACTION_INDEX] = legacy_action_mask[LEGACY_HOLD_ACTION_INDEX] != 0;

    for legacy_action_index in 0..LEGACY_NUM_PLACEMENT_ACTIONS {
        let Some(action_index) =
            action_space.legacy_action_to_index(current_piece, legacy_action_index)
        else {
            continue;
        };
        policy[action_index] += legacy_policy[legacy_action_index];
        action_mask[action_index] |= legacy_action_mask[legacy_action_index] != 0;
    }

    Ok((policy, action_mask))
}

fn argmax_index(values: &[f32]) -> usize {
    let mut max_index = 0;
    let mut max_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > max_value {
            max_value = *value;
            max_index = index;
        }
    }
    max_index
}

fn one_hot<const N: usize>(index: Option<usize>) -> [f32; N] {
    let mut values = [0.0f32; N];
    if let Some(index) = index {
        values[index] = 1.0;
    }
    values
}

fn validate_shape_with_dynamic_batch(
    name: &str,
    actual: &[u64],
    expected_with_batch_placeholder: &[u64],
) -> Result<usize, String> {
    if actual.len() != expected_with_batch_placeholder.len() {
        return Err(format!(
            "{} has rank {}, expected rank {}",
            name,
            actual.len(),
            expected_with_batch_placeholder.len()
        ));
    }
    for i in 1..actual.len() {
        if actual[i] != expected_with_batch_placeholder[i] {
            return Err(format!(
                "{} has shape {:?}, expected {}",
                name,
                actual,
                format_expected_shape(expected_with_batch_placeholder)
            ));
        }
    }
    Ok(actual[0] as usize)
}

fn format_expected_shape(expected_with_batch_placeholder: &[u64]) -> String {
    let dims = expected_with_batch_placeholder
        .iter()
        .enumerate()
        .map(|(index, dim)| {
            if index == 0 {
                "N".to_string()
            } else {
                dim.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{}]", dims)
}

fn open_npy<'a>(
    archive: &'a mut Archive,
    field_name: &str,
) -> Result<NpyFile<BufReader<ZipFile<'a>>>, String> {
    let entry = archive
        .by_name(&format!("{field_name}.npy"))
        .map_err(|e| format!("{field_name}: {e}"))?;
    NpyFile::new(BufReader::new(entry)).map_err(|e| format!("{field_name}: {e}"))
}

fn npy_shape(archive: &mut Archive, field_name: &str) -> Result<Vec<u64>, String> {
    Ok(open_npy(archive, field_name)?.shape().to_vec())
}

fn check_batch_shape(
    field_name: &str,
    shape: &[u64],
    n: usize,
    trailing_dims: &[u64],
) -> Result<(), String> {
    let mut expected = vec![n as u64];
    expected.extend_from_slice(trailing_dims);
    if shape != expected.as_slice() {
        return Err(format!(
            "{} has shape {:?}, expected {:?}",
            field_name, shape, expected
        ));
    }
    Ok(())
}

/// Stream `field_name` row by row (row = product of `trailing_dims` elements).
fn for_each_row<T: npyz::Deserialize + Default + Clone>(
    archive: &mut Archive,
    field_name: &str,
    n: usize,
    trailing_dims: &[u64],
    mut visit: impl FnMut(usize, &[T]),
) -> Result<(), String> {
    let npy = open_npy(archive, field_name)?;
    check_batch_shape(field_name, npy.shape(), n, trailing_dims)?;
    let width = trailing_dims.iter().product::<u64>() as usize;
    let reader = npy.data::<T>().map_err(|e| format!("{field_name}: {e}"))?;
    stream_rows(reader, field_name, width, &mut visit)
}

fn stream_rows<T: Default + Clone>(
    values: impl Iterator<Item = std::io::Result<T>>,
    field_name: &str,
    width: usize,
    visit: &mut impl FnMut(usize, &[T]),
) -> Result<(), String> {
    let mut row = vec![T::default(); width];
    let mut filled = 0;
    let mut row_index = 0;
    for value in values {
        row[filled] = value.map_err(|e| format!("{field_name}: {e}"))?;
        filled += 1;
        if filled == width {
            visit(row_index, &row);
            row_index += 1;
            filled = 0;
        }
    }
    Ok(())
}

/// Stream a uint8- or bool-typed field as `u8` rows of `width` elements.
fn for_each_bool_like_row(
    archive: &mut Archive,
    field_name: &str,
    n: usize,
    width: usize,
    mut visit: impl FnMut(usize, &[u8]),
) -> Result<(), String> {
    let trailing: &[u64] = if width == 1 { &[] } else { &[width as u64] };
    let npy = open_npy(archive, field_name)?;
    check_batch_shape(field_name, npy.shape(), n, trailing)?;
    match npy.try_data::<u8>() {
        Ok(reader) => stream_rows(reader, field_name, width, &mut visit),
        Err(npy) => {
            let reader = npy
                .data::<bool>()
                .map_err(|e| format!("{field_name}: {e}"))?
                .map(|value| value.map(u8::from));
            stream_rows(reader, field_name, width, &mut visit)
        }
    }
}

fn for_each_bool_like(
    archive: &mut Archive,
    field_name: &str,
    n: usize,
    mut visit: impl FnMut(usize, bool),
) -> Result<(), String> {
    for_each_bool_like_row(archive, field_name, n, 1, |i, row| visit(i, row[0] != 0))
}

fn write_scalar_field<T: npyz::Serialize + npyz::AutoSerialize>(
    zip: &mut ZipWriter<BufWriter<File>>,
    options: FileOptions,
    name: &str,
    n: u64,
    data: impl Iterator<Item = T>,
) -> Result<(), String> {
    write_flat_field(zip, options, name, n, &[], data)
}

fn write_flat_field<T: npyz::Serialize + npyz::AutoSerialize>(
    zip: &mut ZipWriter<BufWriter<File>>,
    options: FileOptions,
    name: &str,
    n: u64,
    trailing_dims: &[u64],
    data: impl Iterator<Item = T>,
) -> Result<(), String> {
    let mut shape = Vec::with_capacity(trailing_dims.len() + 1);
    shape.push(n);
    shape.extend_from_slice(trailing_dims);
    zip.start_file(name, options)
        .map_err(|e: zip::result::ZipError| e.to_string())?;
    // npyz issues one small write per element; feeding those straight into the
    // deflate encoder made a 5M-example snapshot take tens of minutes.
    let mut buffered = BufWriter::with_capacity(1 << 20, &mut *zip);
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&shape)
        .writer(&mut buffered)
        .begin_nd()
        .map_err(|e: std::io::Error| e.to_string())?;
    writer
        .extend(data)
        .map_err(|e: std::io::Error| e.to_string())?;
    writer.finish().map_err(|e: std::io::Error| e.to_string())?;
    buffered.flush().map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::action_space::{
        get_action_space, HOLD_ACTION_INDEX, LEGACY_NUM_ACTIONS, LEGACY_NUM_PLACEMENT_ACTIONS,
    };
    use crate::game::constants::ROW_FILL_FEATURE_ROWS;
    use crate::runtime::test_utils;
    use crate::search::{TrainingExample, NUM_ACTIONS};
    use std::fs;
    use std::path::PathBuf;

    fn unique_temp_path(name: &str) -> PathBuf {
        test_utils::unique_temp_path("tetris_core", name)
    }

    fn make_example(move_number: u32, hold_piece: usize, combo_raw: u32) -> TrainingExample {
        use crate::inference::normalize_combo_for_feature;

        let mut board = vec![0u8; BOARD_HEIGHT * BOARD_WIDTH];
        board[0] = 1;
        board[42] = 1;

        let mut policy = vec![0.0; NUM_ACTIONS];
        policy[0] = 0.25;
        policy[13] = 0.75;

        let mut action_mask = vec![false; NUM_ACTIONS];
        action_mask[0] = true;
        action_mask[13] = true;

        TrainingExample {
            board,
            current_piece: 2,
            hold_piece,
            hold_available: true,
            next_queue: vec![0, 1, 2, 3, 4],
            move_number,
            placement_count: move_number as f32 / 100.0,
            combo: normalize_combo_for_feature(combo_raw),
            back_to_back: true,
            next_hidden_piece_probs: vec![0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0],
            column_heights: vec![0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
            max_column_height: 0.45,
            row_fill_counts: vec![0.0; ROW_FILL_FEATURE_ROWS],
            total_blocks: 0.01,
            bumpiness: 0.0025,
            holes: 0.05,
            policy,
            value: 3.5,
            action_mask,
            overhang_fields: crate::search::normalize_overhang_fields(17),
            game_number: 123,
            game_total_attack: 37,
        }
    }

    fn compact(examples: &[TrainingExample]) -> Vec<CompactExample> {
        examples.iter().map(CompactExample::from_example).collect()
    }

    #[test]
    fn test_npz_round_trip_preserves_examples() {
        let path = unique_temp_path("roundtrip");
        let examples = compact(&[make_example(0, 7, 3), make_example(88, 5, 3)]);

        write_examples_to_npz(&path, &examples).expect("write should succeed");
        let loaded = read_examples_from_npz(&path).expect("read should succeed");
        fs::remove_file(&path).expect("temp file cleanup should succeed");

        assert_eq!(loaded, examples);
    }

    #[test]
    fn test_npz_round_trip_keeps_short_queue() {
        let path = unique_temp_path("short_queue");
        let mut example = make_example(3, 7, 0);
        example.next_queue = vec![4, 1];
        let examples = compact(&[example]);

        write_examples_to_npz(&path, &examples).expect("write should succeed");
        let loaded = read_examples_from_npz(&path).expect("read should succeed");
        fs::remove_file(&path).expect("temp file cleanup should succeed");

        assert_eq!(loaded[0].next_queue_pieces(), vec![4, 1]);
    }

    #[test]
    fn test_validate_shape_with_dynamic_batch_rejects_mismatched_dims() {
        let err = validate_shape_with_dynamic_batch("boards", &[2, 19, 10], &[0, 20, 10])
            .expect_err("shape mismatch should error");
        assert!(err.contains("expected [N, 20, 10]"));
    }

    #[test]
    fn test_npz_combo_round_trip_preserves_normalized_value() {
        use crate::inference::normalize_combo_for_feature;

        let path = unique_temp_path("combo_normalized");
        // combo normalization is uncapped linear scaling.
        let examples = compact(&[make_example(0, 7, 99)]);

        write_examples_to_npz(&path, &examples).expect("write should succeed");
        let loaded = read_examples_from_npz(&path).expect("read should succeed");
        fs::remove_file(&path).expect("temp file cleanup should succeed");

        assert_eq!(loaded.len(), 1);
        assert!((loaded[0].combo - normalize_combo_for_feature(99)).abs() < 1e-6);
    }

    #[test]
    fn test_npz_slices_round_trip() {
        let path = unique_temp_path("slices_roundtrip");
        let examples_a = compact(&[make_example(0, 7, 3)]);
        let examples_b = compact(&[make_example(88, 5, 3)]);

        write_examples_slices_to_npz(&path, &examples_a, &examples_b)
            .expect("write should succeed");
        let loaded = read_examples_from_npz(&path).expect("read should succeed");
        fs::remove_file(&path).expect("temp file cleanup should succeed");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].move_number, 0);
        assert_eq!(loaded[1].move_number, 88);
    }

    /// Dense snapshots written before the compact format must still load.
    #[test]
    fn test_reads_dense_policy_snapshot() {
        let path = unique_temp_path("dense_snapshot");
        let examples = vec![make_example(0, 7, 3), make_example(9, 2, 1)];
        write_dense_npz(&path, &examples);
        let loaded = read_examples_from_npz(&path).expect("dense read should succeed");
        fs::remove_file(&path).expect("temp file cleanup should succeed");

        assert_eq!(loaded, compact(&examples));
    }

    fn write_dense_npz(path: &Path, examples: &[TrainingExample]) {
        write_examples_to_npz(path, &compact(examples)).expect("write should succeed");
        // Rewrite the archive with dense policy_targets/action_masks instead of the compact fields.
        let source = fs::read(path).expect("read back");
        let mut reader = ZipArchive::new(std::io::Cursor::new(source)).expect("zip");
        let out = File::create(path).expect("create");
        let mut zip = ZipWriter::new(BufWriter::new(out));
        let options = FileOptions::default().compression_method(CompressionMethod::Stored);
        for index in 0..reader.len() {
            let mut entry = reader.by_index(index).expect("entry");
            let name = entry.name().to_string();
            if name == "policy_legal.npy" || name == "action_mask_bits.npy" {
                continue;
            }
            let mut bytes = Vec::new();
            std::io::copy(&mut entry, &mut bytes).expect("copy");
            zip.start_file(name, options).expect("start");
            std::io::Write::write_all(&mut zip, &bytes).expect("write");
        }
        let n = examples.len() as u64;
        write_flat_field(
            &mut zip,
            options,
            "policy_targets.npy",
            n,
            &[NUM_ACTIONS as u64],
            examples.iter().flat_map(|ex| ex.policy.iter().copied()),
        )
        .expect("policy");
        write_flat_field(
            &mut zip,
            options,
            "action_masks.npy",
            n,
            &[NUM_ACTIONS as u64],
            examples
                .iter()
                .flat_map(|ex| ex.action_mask.iter().map(|&m| m as u8)),
        )
        .expect("masks");
        zip.finish().expect("finish");
    }

    #[test]
    fn test_adapt_legacy_policy_and_mask_collapses_redundant_rotations() {
        let action_space = get_action_space();
        let piece_type = crate::game::constants::I_PIECE;
        let source_actions = (0..LEGACY_NUM_PLACEMENT_ACTIONS)
            .filter_map(|legacy_action_index| {
                action_space
                    .legacy_action_to_index(piece_type, legacy_action_index)
                    .map(|target_action| (legacy_action_index, target_action))
            })
            .collect::<Vec<_>>();
        let target_action = source_actions
            .iter()
            .map(|(_, target_action)| *target_action)
            .find(|candidate_action| {
                source_actions
                    .iter()
                    .filter(|(_, target_action)| target_action == candidate_action)
                    .count()
                    == 2
            })
            .expect("expected duplicated canonical I-piece mapping");
        let source_actions = source_actions
            .into_iter()
            .filter_map(|(legacy_action_index, candidate_action)| {
                (candidate_action == target_action).then_some(legacy_action_index)
            })
            .collect::<Vec<_>>();
        assert_eq!(source_actions.len(), 2);

        let mut legacy_policy = vec![0.0; LEGACY_NUM_ACTIONS];
        legacy_policy[source_actions[0]] = 0.25;
        legacy_policy[source_actions[1]] = 0.75;
        legacy_policy[LEGACY_HOLD_ACTION_INDEX] = 0.5;

        let mut legacy_action_mask = vec![0u8; LEGACY_NUM_ACTIONS];
        legacy_action_mask[source_actions[1]] = 1;
        legacy_action_mask[LEGACY_HOLD_ACTION_INDEX] = 1;

        let (policy, action_mask) = adapt_legacy_policy_and_mask(
            action_space,
            piece_type,
            &legacy_policy,
            &legacy_action_mask,
        )
        .expect("legacy adapter should succeed");

        assert_eq!(policy.len(), NUM_ACTIONS);
        assert_eq!(action_mask.len(), NUM_ACTIONS);
        assert!((policy[target_action] - 1.0).abs() < 1e-6);
        assert!((policy[HOLD_ACTION_INDEX] - 0.5).abs() < 1e-6);
        assert!(action_mask[target_action]);
        assert!(action_mask[HOLD_ACTION_INDEX]);
    }
}
