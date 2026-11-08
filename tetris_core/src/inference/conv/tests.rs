use super::*;
#[cfg(target_arch = "x86_64")]
use crate::inference::kernels::Avx2;
#[cfg(target_arch = "aarch64")]
use crate::inference::kernels::Neon;
use crate::inference::kernels::Portable;
use crate::inference::onnx_proto::{decode_model, OnnxAttribute};

impl ConvTrunk {
    fn forward(&self, board: &[f32], scratch: &mut ConvScratch, out: &mut [f32]) {
        unsafe {
            match self.isa {
                #[cfg(target_arch = "aarch64")]
                Isa::Neon => self.forward_with::<Neon>(board, scratch, out),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2Fma => forward_avx2(self, board, scratch, out),
                Isa::Portable => self.forward_with::<Portable>(board, scratch, out),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn forward_avx2(
    trunk: &ConvTrunk,
    board: &[f32],
    scratch: &mut ConvScratch,
    out: &mut [f32],
) {
    trunk.forward_with::<Avx2>(board, scratch, out)
}

/// Tiny trunk exported with the production exporter (`export_split_models`):
/// `TetrisNet(trunk_channels=8, num_conv_residual_blocks=2, reduction_channels=16, ...)` with
/// `torch.manual_seed(1234)` and every GroupNorm affine / bias perturbed by `0.3 * randn`.
const FIXTURE: &[u8] = include_bytes!("../testdata/tiny_trunk.conv.onnx");

fn fixture_model() -> OnnxModel {
    decode_model(FIXTURE).expect("fixture decodes")
}

fn lcg_values(seed: u64, len: usize, scale: f32) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f64) / ((1u64 << 24) as f64);
            ((unit * 2.0 - 1.0) as f32) * scale
        })
        .collect()
}

fn test_boards() -> Vec<Vec<f32>> {
    let mut boards = vec![vec![0.0; PIXELS], vec![1.0; PIXELS]];
    for (seed, density) in [(1u64, 0.1f32), (2, 0.35), (3, 0.6), (4, 0.9)] {
        let noise = lcg_values(seed, PIXELS, 1.0);
        boards.push(
            noise
                .iter()
                .map(|&v| if (v + 1.0) / 2.0 < density { 1.0 } else { 0.0 })
                .collect(),
        );
    }
    // Typical stacked board: bottom rows mostly full with a well.
    boards.push(
        (0..PIXELS)
            .map(|i| {
                let (row, col) = (i / BOARD_WIDTH, i % BOARD_WIDTH);
                if row >= 14 && col != 7 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
    );
    // Non-binary values must also follow the conv semantics exactly.
    boards.push(lcg_values(9, PIXELS, 1.0));
    boards
}

/// f64 NCHW reference of `ConvBackbone.forward` (flattened like PyTorch).
fn reference_forward(spec: &ConvTrunkSpec, board: &[f32]) -> Vec<f64> {
    let (h, w) = (BOARD_HEIGHT, BOARD_WIDTH);
    let mut input = vec![0.0f64; 3 * PIXELS];
    for i in 0..PIXELS {
        input[i] = f64::from(board[i]);
        input[PIXELS + i] = f64::from(spec.row_coords[i]);
        input[2 * PIXELS + i] = f64::from(spec.col_coords[i]);
    }
    let conv = |x: &[f64], hh: usize, ww: usize, c: &ConvSpec| -> (Vec<f64>, usize) {
        let oh = (hh - 1) / c.stride + 1;
        let ow = (ww - 1) / c.stride + 1;
        let mut out = vec![0.0; c.out_channels * oh * ow];
        for co in 0..c.out_channels {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = f64::from(c.bias[co]);
                    for ci in 0..c.in_channels {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let iy = (oy * c.stride + ky) as isize - 1;
                                let ix = (ox * c.stride + kx) as isize - 1;
                                if iy < 0 || ix < 0 || iy >= hh as isize || ix >= ww as isize {
                                    continue;
                                }
                                acc += x[(ci * hh + iy as usize) * ww + ix as usize]
                                    * f64::from(
                                        c.weight[((co * c.in_channels + ci) * 3 + ky) * 3 + kx],
                                    );
                            }
                        }
                    }
                    out[(co * oh + oy) * ow + ox] = acc;
                }
            }
        }
        (out, oh * ow)
    };
    let gn_silu = |x: &mut [f64], n: &NormSpec, pixels: usize| {
        let group = n.channels / n.groups * pixels;
        for g in 0..n.groups {
            let range = g * group..(g + 1) * group;
            let mean = x[range.clone()].iter().sum::<f64>() / group as f64;
            let var = x[range.clone()]
                .iter()
                .map(|v| (v - mean) * (v - mean))
                .sum::<f64>()
                / group as f64;
            for idx in range {
                let ch = idx / pixels;
                let y = (x[idx] - mean) / (var + f64::from(n.epsilon)).sqrt()
                    * f64::from(n.gamma[ch])
                    + f64::from(n.beta[ch]);
                x[idx] = y / (1.0 + (-y).exp());
            }
        }
    };
    let (mut x, _) = conv(&input, h, w, &spec.stem);
    gn_silu(&mut x, &spec.stem_norm, PIXELS);
    for block in &spec.blocks {
        let mut t = x.clone();
        gn_silu(&mut t, &block.norm1, PIXELS);
        let (mut t, _) = conv(&t, h, w, &block.conv1);
        gn_silu(&mut t, &block.norm2, PIXELS);
        let (t, _) = conv(&t, h, w, &block.conv2);
        for (xi, ti) in x.iter_mut().zip(t) {
            *xi += ti;
        }
    }
    let (mut r, pixels) = conv(&x, h, w, &spec.reduce);
    gn_silu(&mut r, &spec.reduce_norm, pixels);
    r
}

/// Max abs diff between the native trunk (any ISA) and the f64 reference, over `boards`.
fn max_abs_diff(spec: &ConvTrunkSpec, isa: Isa, boards: &[Vec<f32>]) -> f64 {
    let trunk = ConvTrunk::from_spec(spec, isa);
    let mut scratch = trunk.new_scratch();
    let mut out = vec![0.0f32; trunk.output_size()];
    let mut worst = 0.0f64;
    for board in boards {
        trunk.forward(board, &mut scratch, &mut out);
        let expected = reference_forward(spec, board);
        for (k, &got) in out.iter().enumerate() {
            let want = expected[trunk.feature_source_index(k)];
            worst = worst.max((f64::from(got) - want).abs() / (1.0 + want.abs()));
        }
    }
    worst
}

fn random_conv(seed: u64, cin: usize, cout: usize, stride: usize) -> ConvSpec {
    let scale = (2.0 / (9.0 * cin as f32)).sqrt();
    ConvSpec {
        in_channels: cin,
        out_channels: cout,
        stride,
        weight: lcg_values(seed, cout * cin * 9, scale),
        bias: lcg_values(seed + 1, cout, 0.2),
    }
}

fn random_norm(seed: u64, channels: usize, groups: usize) -> NormSpec {
    NormSpec {
        channels,
        groups,
        epsilon: 1e-5,
        gamma: lcg_values(seed, channels, 0.5)
            .iter()
            .map(|v| v + 1.0)
            .collect(),
        beta: lcg_values(seed + 1, channels, 0.3),
    }
}

fn random_spec(channels: usize, reduce: usize, blocks: usize, groups: usize) -> ConvTrunkSpec {
    ConvTrunkSpec {
        row_coords: (0..PIXELS)
            .map(|i| (i / BOARD_WIDTH) as f32 / (BOARD_HEIGHT - 1) as f32)
            .collect(),
        col_coords: (0..PIXELS)
            .map(|i| (i % BOARD_WIDTH) as f32 / (BOARD_WIDTH - 1) as f32)
            .collect(),
        stem: random_conv(100, 3, channels, 1),
        stem_norm: random_norm(200, channels, groups),
        blocks: (0..blocks as u64)
            .map(|b| ResidualBlockSpec {
                norm1: random_norm(300 + 10 * b, channels, groups),
                conv1: random_conv(400 + 10 * b, channels, channels, 1),
                norm2: random_norm(500 + 10 * b, channels, groups),
                conv2: random_conv(600 + 10 * b, channels, channels, 1),
            })
            .collect(),
        reduce: random_conv(700, channels, reduce, 2),
        reduce_norm: random_norm(800, reduce, reduce.min(groups)),
    }
}

#[test]
fn native_trunk_matches_reference_for_random_weights() {
    let boards = test_boards();
    for (channels, reduce, blocks, groups) in [(32, 32, 5, 32), (16, 8, 2, 16), (32, 16, 1, 8)] {
        let spec = random_spec(channels, reduce, blocks, groups);
        for isa in Isa::available() {
            let diff = max_abs_diff(&spec, isa, &boards);
            assert!(
                diff < 2e-5,
                "{isa:?} c={channels} r={reduce} blocks={blocks}: max rel diff {diff}"
            );
        }
    }
}

#[test]
fn fixture_parses_into_expected_structure() {
    let spec = ConvTrunkSpec::from_onnx_model(&fixture_model()).expect("fixture is valid");
    assert_eq!(spec.channels(), 8);
    assert_eq!(spec.blocks.len(), 2);
    assert_eq!(spec.reduce.out_channels, 16);
    assert_eq!(spec.output_size(), 16 * REDUCED_PIXELS);
    assert_eq!(spec.stem_norm.groups, 8);
    assert_eq!(spec.reduce_norm.groups, 16);
    assert_eq!(spec.stem_norm.epsilon, 1e-5);
    // CoordConv planes: linspace(0, 1) over rows / columns.
    assert_eq!(spec.row_coords[0], 0.0);
    assert_eq!(spec.row_coords[PIXELS - 1], 1.0);
    assert_eq!(spec.col_coords[BOARD_WIDTH - 1], 1.0);
    assert_eq!(spec.col_coords[BOARD_WIDTH], 0.0);
    // The perturbed GroupNorm affines must come through (not the 1/0 InstanceNorm constants).
    assert!(spec.blocks[1]
        .norm2
        .gamma
        .iter()
        .any(|&g| (g - 1.0).abs() > 1e-3));
    for isa in Isa::available() {
        let diff = max_abs_diff(&spec, isa, &test_boards());
        assert!(diff < 2e-5, "{isa:?}: fixture max rel diff {diff}");
    }
}

/// Real-model check, e.g.
/// `TETRIS_TEST_CONV_ONNX=.../checkpoints/incumbent.conv.onnx cargo test --release incumbent`.
#[test]
fn incumbent_conv_onnx_matches_reference_when_available() {
    let Ok(path) = std::env::var("TETRIS_TEST_CONV_ONNX") else {
        eprintln!("Skipping - set TETRIS_TEST_CONV_ONNX to a bundle's .conv.onnx");
        return;
    };
    let spec = ConvTrunkSpec::from_onnx_path(Path::new(&path)).expect("incumbent conv loads");
    eprintln!(
        "{path}: channels={} blocks={} reduce={} groups={} eps={} output={}",
        spec.channels(),
        spec.blocks.len(),
        spec.reduce.out_channels,
        spec.stem_norm.groups,
        spec.stem_norm.epsilon,
        spec.output_size()
    );
    for isa in Isa::available() {
        let diff = max_abs_diff(&spec, isa, &test_boards());
        eprintln!("{isa:?}: max rel diff vs f64 reference {diff:e}");
        assert!(diff < 2e-5, "{isa:?}: incumbent max rel diff {diff}");
    }
}

fn expect_rejected(model: &OnnxModel, needle: &str) {
    match ConvTrunkSpec::from_onnx_model(model) {
        Ok(_) => panic!("mutated graph was accepted (expected error containing '{needle}')"),
        Err(e) => assert!(
            e.to_string().contains(needle),
            "error '{e}' does not mention '{needle}'"
        ),
    }
}

fn nth_node<'a>(model: &'a mut OnnxModel, op: &str, n: usize) -> &'a mut OnnxNode {
    model
        .graph
        .nodes
        .iter_mut()
        .filter(|node| node.op_type == op)
        .nth(n)
        .unwrap_or_else(|| panic!("fixture has no {op} #{n}"))
}

fn set_attr(node: &mut OnnxNode, name: &str, value: OnnxAttributeValue) {
    match node.attributes.iter_mut().find(|a| a.name == name) {
        Some(attr) => attr.value = value,
        None => node.attributes.push(OnnxAttribute {
            name: name.to_string(),
            value,
        }),
    }
}

/// The tensor of the `Constant` node producing `consumer_input`.
fn constant_feeding<'a>(model: &'a mut OnnxModel, consumer_input: &str) -> &'a mut OnnxTensor {
    let node = model
        .graph
        .nodes
        .iter_mut()
        .find(|node| node.op_type == "Constant" && node.outputs[0] == consumer_input)
        .expect("constant producer");
    match &mut node.attributes[0].value {
        OnnxAttributeValue::Tensor(tensor) => tensor,
        other => panic!("constant value is {other:?}"),
    }
}

#[test]
fn strict_validation_rejects_graph_mutations() {
    let mut m = fixture_model();
    set_attr(
        nth_node(&mut m, "Conv", 1),
        "strides",
        OnnxAttributeValue::Ints(vec![2, 2]),
    );
    expect_rejected(&m, "expected stride-1 Conv");

    let mut m = fixture_model();
    set_attr(
        nth_node(&mut m, "Conv", 2),
        "pads",
        OnnxAttributeValue::Ints(vec![0, 0, 0, 0]),
    );
    expect_rejected(&m, "pads");

    let mut m = fixture_model();
    set_attr(
        nth_node(&mut m, "Conv", 0),
        "group",
        OnnxAttributeValue::Int(2),
    );
    expect_rejected(&m, "grouped Conv");

    let mut m = fixture_model();
    set_attr(
        nth_node(&mut m, "Conv", 3),
        "dilations",
        OnnxAttributeValue::Ints(vec![2, 2]),
    );
    expect_rejected(&m, "dilations");

    let mut m = fixture_model();
    set_attr(
        nth_node(&mut m, "Conv", 0),
        "foo",
        OnnxAttributeValue::Int(1),
    );
    expect_rejected(&m, "unexpected attribute 'foo'");

    let mut m = fixture_model();
    nth_node(&mut m, "Sigmoid", 2).op_type = "Relu".to_string();
    expect_rejected(&m, "unsupported op 'Relu'");

    let mut m = fixture_model();
    nth_node(&mut m, "Add", 2).domain = "com.microsoft".to_string();
    expect_rejected(&m, "unsupported op domain");

    let mut m = fixture_model();
    set_attr(
        nth_node(&mut m, "InstanceNormalization", 1),
        "epsilon",
        OnnxAttributeValue::Float(-1.0),
    );
    expect_rejected(&m, "epsilon");

    let mut m = fixture_model();
    set_attr(
        nth_node(&mut m, "InstanceNormalization", 3),
        "epsilon",
        OnnxAttributeValue::Int(1),
    );
    expect_rejected(&m, "invalid epsilon");

    // InstanceNormalization must have unit scale / zero bias (the affine is the Mul/Add pair).
    let mut m = fixture_model();
    let scale_name = nth_node(&mut m, "InstanceNormalization", 0).inputs[1].clone();
    constant_feeding(&mut m, &scale_name).data = OnnxTensorData::Float(vec![2.0; 8]);
    expect_rejected(&m, "unit scale");

    // GroupNorm grouping reshape with a group count that does not divide the channels.
    let mut m = fixture_model();
    let shape_name = nth_node(&mut m, "Reshape", 0).inputs[1].clone();
    constant_feeding(&mut m, &shape_name).data = OnnxTensorData::Int64(vec![0, 3, -1]);
    let norm_inputs = nth_node(&mut m, "InstanceNormalization", 0).inputs.clone();
    for (input, fill) in [(&norm_inputs[1], 1.0f32), (&norm_inputs[2], 0.0)] {
        let tensor = constant_feeding(&mut m, input);
        tensor.dims = vec![3];
        tensor.data = OnnxTensorData::Float(vec![fill; 3]);
    }
    expect_rejected(&m, "do not divide");

    // Final flatten replaced by a different reshape.
    let mut m = fixture_model();
    let reshapes = m
        .graph
        .nodes
        .iter()
        .filter(|n| n.op_type == "Reshape")
        .count();
    let flatten_shape = nth_node(&mut m, "Reshape", reshapes - 1).inputs[1].clone();
    constant_feeding(&mut m, &flatten_shape).data = OnnxTensorData::Int64(vec![1, 16, -1]);
    expect_rejected(&m, "unsupported Reshape target");

    // Board plane must come first (swapping the two coordinate planes is a valid graph).
    let mut m = fixture_model();
    nth_node(&mut m, "Concat", 0).inputs.swap(0, 1);
    expect_rejected(&m, "Concat must be");

    // Residual connection removed (Add replaced by Mul of two activations).
    let mut m = fixture_model();
    let residual_add = m
        .graph
        .nodes
        .iter_mut()
        .filter(|n| n.op_type == "Add")
        .find(|n| n.outputs[0].contains("res_blocks.1/Add"))
        .expect("residual add");
    residual_add.op_type = "Mul".to_string();
    expect_rejected(&m, "x * Sigmoid(x)");

    // GroupNorm affine with a non-broadcast-safe shape.
    let mut m = fixture_model();
    let initializer_names: Vec<String> = m
        .graph
        .initializers
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let gamma_name = m
        .graph
        .nodes
        .iter()
        .find(|n| n.op_type == "Mul" && initializer_names.contains(&n.inputs[1]))
        .map(|n| n.inputs[1].clone())
        .expect("GroupNorm gamma Mul");
    let gamma = m
        .graph
        .initializers
        .iter_mut()
        .find(|t| t.name == gamma_name)
        .expect("gamma initializer");
    gamma.dims = vec![8];
    expect_rejected(&m, "GroupNorm gamma dims");

    // Wrong runtime input shape and opset.
    let mut m = fixture_model();
    m.graph.inputs[0].dims[3] = OnnxDim::Value(11);
    expect_rejected(&m, "has dims");
    let mut m = fixture_model();
    m.opset_version = 9;
    expect_rejected(&m, "unsupported ONNX opset");

    // Wrong output width (graph output declares a different feature count).
    let mut m = fixture_model();
    m.graph.outputs[0].dims[1] = OnnxDim::Value(1600);
    expect_rejected(&m, "has dims");
}

#[test]
fn identity_aliases_are_resolved() {
    // Fresh models export identical GroupNorm affines once and alias them through Identity.
    let mut m = fixture_model();
    let initializer_names: Vec<String> = m
        .graph
        .initializers
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let mul_index = m
        .graph
        .nodes
        .iter()
        .position(|n| n.op_type == "Mul" && initializer_names.contains(&n.inputs[1]))
        .expect("GroupNorm gamma Mul");
    let gamma_name = m.graph.nodes[mul_index].inputs[1].clone();
    m.graph.nodes[mul_index].inputs[1] = "gamma_alias".to_string();
    m.graph.nodes.insert(
        mul_index,
        OnnxNode {
            op_type: "Identity".to_string(),
            inputs: vec![gamma_name],
            outputs: vec!["gamma_alias".to_string()],
            ..OnnxNode::default()
        },
    );
    let aliased = ConvTrunkSpec::from_onnx_model(&m).expect("Identity alias is valid");
    let original = ConvTrunkSpec::from_onnx_model(&fixture_model()).unwrap();
    assert_eq!(aliased.stem_norm.gamma, original.stem_norm.gamma);

    m.graph.nodes[mul_index].attributes.push(OnnxAttribute {
        name: "bogus".to_string(),
        value: OnnxAttributeValue::Int(0),
    });
    expect_rejected(&m, "unexpected attribute 'bogus'");
}

#[test]
fn accepted_epsilon_is_propagated_per_layer() {
    let mut m = fixture_model();
    set_attr(
        nth_node(&mut m, "InstanceNormalization", 2),
        "epsilon",
        OnnxAttributeValue::Float(1e-3),
    );
    let spec = ConvTrunkSpec::from_onnx_model(&m).expect("valid epsilon");
    // InstanceNorm #2 is block 0's second GroupNorm (stem, block0.norm1, block0.norm2, ...).
    assert_eq!(spec.blocks[0].norm2.epsilon, 1e-3);
    assert_eq!(spec.blocks[0].norm1.epsilon, 1e-5);
}

#[test]
fn truncated_onnx_bytes_fail_to_decode() {
    for len in [10, 1000, FIXTURE.len() - 7] {
        assert!(
            decode_model(&FIXTURE[..len]).is_err(),
            "prefix of {len} bytes decoded"
        );
    }
}
