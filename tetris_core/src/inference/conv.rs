//! Native conv trunk for the cached board-embedding path.
//!
//! Weights come straight from the exported `X.conv.onnx`, so every existing bundle works
//! without re-export. The graph is interpreted symbolically and must match the exact
//! `ConvBackbone` structure from `tetris_bot/ml/network.py`:
//!
//! ```text
//! concat(board, row_coords, col_coords) -> conv3x3 -> GN -> SiLU            (stem)
//! N x [x + conv3x3(SiLU(GN(conv3x3(SiLU(GN(x))))))]                         (pre-act blocks)
//! conv3x3 stride 2 -> GN -> SiLU -> flatten                                 (reduction)
//! ```
//!
//! with GroupNorm exported as `Reshape[0,G,-1] -> InstanceNormalization(1, 0) -> Reshape(Shape)
//! -> Mul(gamma) -> Add(beta)` and SiLU as `x * Sigmoid(x)`. Any other op, attribute, or shape
//! fails loudly at load time.
//!
//! The forward pass runs channels-last. The two coordinate channels are constant, so their
//! contribution plus the stem bias is precomputed as a per-pixel bias map and the binary board
//! channel is applied as a sparse scatter. Output features are channels-last (`[pixel][channel]`);
//! [`ConvTrunk::feature_source_index`] maps them back to PyTorch's flattened NCHW order.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use super::kernels::{
    copy_image, group_norm_silu, ConvKernel, ConvPixelTable, GroupNormParams, ImageLayout, Isa,
    PackedConv3x3, Simd, CONV_NARROW_CHANNEL_TILE, MAX_NORM_CHANNELS,
};
use super::onnx_proto::{
    self, invalid_data, OnnxAttributeValue, OnnxDim, OnnxGraph, OnnxModel, OnnxNode, OnnxTensor,
    OnnxTensorData, ONNX_ELEM_FLOAT,
};
use crate::game::constants::{BOARD_HEIGHT, BOARD_WIDTH};

const SUPPORTED_OPSETS: std::ops::RangeInclusive<i64> = 13..=21;
const PIXELS: usize = BOARD_HEIGHT * BOARD_WIDTH;
const REDUCED_HEIGHT: usize = (BOARD_HEIGHT + 1) / 2;
const REDUCED_WIDTH: usize = (BOARD_WIDTH + 1) / 2;
const REDUCED_PIXELS: usize = REDUCED_HEIGHT * REDUCED_WIDTH;
const PADDED_HEIGHT: usize = BOARD_HEIGHT + 2;
const PADDED_WIDTH: usize = BOARD_WIDTH + 2;

// ---------------------------------------------------------------------------------------------
// Validated trunk definition (PyTorch layouts)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct ConvSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub stride: usize,
    /// `[out][in][3][3]`.
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct NormSpec {
    pub channels: usize,
    pub groups: usize,
    pub epsilon: f32,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct ResidualBlockSpec {
    pub norm1: NormSpec,
    pub conv1: ConvSpec,
    pub norm2: NormSpec,
    pub conv2: ConvSpec,
}

#[derive(Clone, Debug)]
pub(crate) struct ConvTrunkSpec {
    /// `[H*W]` coordinate planes concatenated after the board channel.
    pub row_coords: Vec<f32>,
    pub col_coords: Vec<f32>,
    pub stem: ConvSpec,
    pub stem_norm: NormSpec,
    pub blocks: Vec<ResidualBlockSpec>,
    pub reduce: ConvSpec,
    pub reduce_norm: NormSpec,
}

impl ConvTrunkSpec {
    pub(crate) fn from_onnx_path(path: &Path) -> io::Result<Self> {
        let model = onnx_proto::load_model(path)?;
        Self::from_onnx_model(&model).map_err(|e| invalid_data(format!("{}: {e}", path.display())))
    }

    pub(crate) fn from_onnx_model(model: &OnnxModel) -> io::Result<Self> {
        if !SUPPORTED_OPSETS.contains(&model.opset_version) {
            return err(format!(
                "unsupported ONNX opset {} (supported {:?})",
                model.opset_version, SUPPORTED_OPSETS
            ));
        }
        interpret_graph(&model.graph)
            .map_err(|e| invalid_data(format!("not a supported conv trunk graph: {e}")))
    }

    pub(crate) fn channels(&self) -> usize {
        self.stem.out_channels
    }

    pub(crate) fn output_size(&self) -> usize {
        self.reduce.out_channels * REDUCED_PIXELS
    }
}

// ---------------------------------------------------------------------------------------------
// Symbolic graph interpretation
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Const {
    F32 { dims: Vec<i64>, data: Vec<f32> },
    I64 { dims: Vec<i64>, data: Vec<i64> },
    Bool { dims: Vec<i64>, data: Vec<bool> },
}

impl Const {
    fn from_tensor(tensor: &OnnxTensor) -> Self {
        match &tensor.data {
            OnnxTensorData::Float(data) => Const::F32 {
                dims: tensor.dims.clone(),
                data: data.clone(),
            },
            OnnxTensorData::Int64(data) => Const::I64 {
                dims: tensor.dims.clone(),
                data: data.clone(),
            },
        }
    }

    fn dims(&self) -> &[i64] {
        match self {
            Const::F32 { dims, .. } | Const::I64 { dims, .. } | Const::Bool { dims, .. } => dims,
        }
    }

    fn len(&self) -> usize {
        match self {
            Const::F32 { data, .. } => data.len(),
            Const::I64 { data, .. } => data.len(),
            Const::Bool { data, .. } => data.len(),
        }
    }
}

#[derive(Clone, Debug)]
enum Value {
    Const(Const),
    Act(usize),
    ShapeOf(usize),
}

#[derive(Clone, Debug)]
enum Act {
    Board,
    BoardWithCoords {
        row: Vec<f32>,
        col: Vec<f32>,
    },
    Conv {
        input: usize,
        spec: ConvSpec,
    },
    GnGrouped {
        input: usize,
        groups: usize,
    },
    GnInstanceNorm {
        input: usize,
        groups: usize,
        epsilon: f32,
    },
    GnRestored {
        input: usize,
        groups: usize,
        epsilon: f32,
    },
    GnScaled {
        input: usize,
        groups: usize,
        epsilon: f32,
        gamma: Vec<f32>,
    },
    GroupNorm {
        input: usize,
        norm: NormSpec,
    },
    Sigmoid {
        input: usize,
    },
    Silu {
        input: usize,
    },
    Add {
        lhs: usize,
        rhs: usize,
    },
    Flatten {
        input: usize,
    },
}

struct Interpreter {
    values: HashMap<String, Value>,
    acts: Vec<Act>,
}

fn err<T>(message: impl Into<String>) -> io::Result<T> {
    Err(invalid_data(message))
}

fn interpret_graph(graph: &OnnxGraph) -> io::Result<ConvTrunkSpec> {
    let mut interp = Interpreter {
        values: HashMap::new(),
        acts: Vec::new(),
    };
    for tensor in &graph.initializers {
        interp.define(&tensor.name, Value::Const(Const::from_tensor(tensor)))?;
    }

    let runtime_inputs: Vec<_> = graph
        .inputs
        .iter()
        .filter(|input| !interp.values.contains_key(&input.name))
        .collect();
    let [board_input] = runtime_inputs.as_slice() else {
        return err(format!(
            "expected exactly one runtime input, found {}",
            runtime_inputs.len()
        ));
    };
    check_io_dims(
        &board_input.name,
        board_input.elem_type,
        &board_input.dims,
        &[1, BOARD_HEIGHT as i64, BOARD_WIDTH as i64],
    )?;
    let board = interp.push(Act::Board);
    interp.define(&board_input.name, Value::Act(board))?;

    for node in &graph.nodes {
        interp
            .step(node)
            .map_err(|e| invalid_data(format!("node {} {:?}: {e}", node.op_type, node.outputs)))?;
    }

    let [output] = graph.outputs.as_slice() else {
        return err(format!(
            "expected exactly one graph output, found {}",
            graph.outputs.len()
        ));
    };
    let Some(Value::Act(output_act)) = interp.values.get(&output.name).cloned() else {
        return err(format!(
            "graph output '{}' is not a computed activation",
            output.name
        ));
    };
    let spec = match_trunk(&interp.acts, output_act)?;
    check_io_dims(
        &output.name,
        output.elem_type,
        &output.dims,
        &[spec.output_size() as i64],
    )?;
    Ok(spec)
}

/// Runtime I/O must be float32 with a batch dim (fixed 1 or symbolic) followed by `expected`.
fn check_io_dims(name: &str, elem_type: i64, dims: &[OnnxDim], expected: &[i64]) -> io::Result<()> {
    if elem_type != ONNX_ELEM_FLOAT {
        return err(format!(
            "'{name}' must be float32, got elem_type {elem_type}"
        ));
    }
    let batch_ok = matches!(
        dims.first(),
        Some(OnnxDim::Value(1)) | Some(OnnxDim::Param(_))
    );
    let rest_ok = dims.len() == expected.len() + 1
        && dims[1..]
            .iter()
            .zip(expected)
            .all(|(dim, &want)| *dim == OnnxDim::Value(want));
    if !batch_ok || !rest_ok {
        return err(format!(
            "'{name}' has dims {dims:?}, expected [batch, {expected:?}]"
        ));
    }
    Ok(())
}

fn check_attributes(node: &OnnxNode, allowed: &[&str]) -> io::Result<()> {
    for attr in &node.attributes {
        if !allowed.contains(&attr.name.as_str()) {
            return err(format!("unexpected attribute '{}'", attr.name));
        }
    }
    Ok(())
}

fn int_attr(node: &OnnxNode, name: &str) -> io::Result<Option<i64>> {
    match node.attribute(name) {
        None => Ok(None),
        Some(OnnxAttributeValue::Int(value)) => Ok(Some(*value)),
        Some(other) => err(format!("attribute '{name}' must be an int, got {other:?}")),
    }
}

fn ints_attr(node: &OnnxNode, name: &str) -> io::Result<Option<Vec<i64>>> {
    match node.attribute(name) {
        None => Ok(None),
        Some(OnnxAttributeValue::Ints(values)) => Ok(Some(values.clone())),
        Some(other) => err(format!("attribute '{name}' must be ints, got {other:?}")),
    }
}

fn broadcast_dims(a: &[i64], b: &[i64]) -> io::Result<Vec<i64>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0; rank];
    for i in 0..rank {
        let da = if i + a.len() >= rank {
            a[i + a.len() - rank]
        } else {
            1
        };
        let db = if i + b.len() >= rank {
            b[i + b.len() - rank]
        } else {
            1
        };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return err(format!("cannot broadcast {a:?} with {b:?}")),
        };
    }
    Ok(out)
}

/// Element-wise op over small constant tensors; only identical shapes or scalar operands.
fn zip_consts<A: Copy, B: Copy, R>(
    lhs: (&[i64], &[A]),
    rhs: (&[i64], &[B]),
    op: impl Fn(A, B) -> R,
) -> io::Result<(Vec<i64>, Vec<R>)> {
    let dims = broadcast_dims(lhs.0, rhs.0)?;
    let n: usize = dims.iter().map(|&d| d as usize).product();
    let pick = |len: usize, i: usize| if len == 1 { 0 } else { i };
    if (lhs.1.len() != n && lhs.1.len() != 1) || (rhs.1.len() != n && rhs.1.len() != 1) {
        return err("constant broadcasting beyond scalars is not supported");
    }
    let data = (0..n)
        .map(|i| op(lhs.1[pick(lhs.1.len(), i)], rhs.1[pick(rhs.1.len(), i)]))
        .collect();
    Ok((dims, data))
}

fn is_channel_vector(dims: &[i64]) -> Option<usize> {
    match dims {
        [c, 1, 1] | [1, c, 1, 1] if *c > 0 => Some(*c as usize),
        _ => None,
    }
}

impl Interpreter {
    fn push(&mut self, act: Act) -> usize {
        self.acts.push(act);
        self.acts.len() - 1
    }

    fn define(&mut self, name: &str, value: Value) -> io::Result<()> {
        if name.is_empty() {
            return err("empty value name");
        }
        if self.values.insert(name.to_string(), value).is_some() {
            return err(format!("value '{name}' is defined twice"));
        }
        Ok(())
    }

    fn input(&self, node: &OnnxNode, index: usize) -> io::Result<Value> {
        let name = node
            .inputs
            .get(index)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid_data(format!("missing input {index}")))?;
        self.values
            .get(name)
            .cloned()
            .ok_or_else(|| invalid_data(format!("input '{name}' is undefined")))
    }

    fn inputs<const N: usize>(&self, node: &OnnxNode) -> io::Result<[Value; N]> {
        if node.inputs.len() != N {
            return err(format!("expected {N} inputs, got {}", node.inputs.len()));
        }
        let mut values = Vec::with_capacity(N);
        for index in 0..N {
            values.push(self.input(node, index)?);
        }
        Ok(values.try_into().expect("length checked above"))
    }

    fn step(&mut self, node: &OnnxNode) -> io::Result<()> {
        if !(node.domain.is_empty() || node.domain == "ai.onnx") {
            return err(format!("unsupported op domain '{}'", node.domain));
        }
        let [output] = node.outputs.as_slice() else {
            return err(format!("expected one output, got {}", node.outputs.len()));
        };
        let value = match node.op_type.as_str() {
            "Constant" => self.op_constant(node)?,
            "ConstantOfShape" => self.op_constant_of_shape(node)?,
            "Mul" => self.op_mul(node)?,
            "Add" => self.op_add(node)?,
            "Equal" => self.op_equal(node)?,
            "Where" => self.op_where(node)?,
            "Expand" => self.op_expand(node)?,
            "Concat" => self.op_concat(node)?,
            "Conv" => self.op_conv(node)?,
            "Reshape" => self.op_reshape(node)?,
            "InstanceNormalization" => self.op_instance_norm(node)?,
            // The exporter aliases deduplicated constants (e.g. identical GroupNorm affines).
            "Identity" => {
                check_attributes(node, &[])?;
                let [value] = self.inputs::<1>(node)?;
                value
            }
            "Shape" => {
                check_attributes(node, &[])?;
                match self.inputs::<1>(node)? {
                    [Value::Act(act)] => Value::ShapeOf(act),
                    _ => return err("Shape of a non-activation"),
                }
            }
            "Sigmoid" => {
                check_attributes(node, &[])?;
                match self.inputs::<1>(node)? {
                    [Value::Act(act)] => Value::Act(self.push(Act::Sigmoid { input: act })),
                    _ => return err("Sigmoid of a non-activation"),
                }
            }
            other => return err(format!("unsupported op '{other}'")),
        };
        self.define(output, value)
    }

    fn op_constant(&mut self, node: &OnnxNode) -> io::Result<Value> {
        check_attributes(node, &["value"])?;
        if !node.inputs.is_empty() {
            return err("Constant takes no inputs");
        }
        match node.attribute("value") {
            Some(OnnxAttributeValue::Tensor(tensor)) => {
                Ok(Value::Const(Const::from_tensor(tensor)))
            }
            _ => err("Constant without a tensor 'value'"),
        }
    }

    fn op_constant_of_shape(&mut self, node: &OnnxNode) -> io::Result<Value> {
        check_attributes(node, &["value"])?;
        let [Value::Const(Const::I64 { dims, data: shape })] = self.inputs::<1>(node)? else {
            return err("ConstantOfShape needs a constant int64 shape");
        };
        if dims.len() != 1 || shape.iter().any(|&d| d < 0) {
            return err(format!("invalid ConstantOfShape shape {shape:?}"));
        }
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let fill = match node.attribute("value") {
            None => Const::F32 {
                dims: vec![1],
                data: vec![0.0],
            },
            Some(OnnxAttributeValue::Tensor(tensor)) if tensor.num_elements() == 1 => {
                Const::from_tensor(tensor)
            }
            Some(other) => return err(format!("invalid ConstantOfShape value {other:?}")),
        };
        Ok(Value::Const(match fill {
            Const::F32 { data, .. } => Const::F32 {
                dims: shape,
                data: vec![data[0]; n],
            },
            Const::I64 { data, .. } => Const::I64 {
                dims: shape,
                data: vec![data[0]; n],
            },
            Const::Bool { .. } => unreachable!("tensors never decode to bool"),
        }))
    }

    fn op_mul(&mut self, node: &OnnxNode) -> io::Result<Value> {
        check_attributes(node, &[])?;
        match self.inputs::<2>(node)? {
            [Value::Const(Const::I64 { dims: da, data: a }), Value::Const(Const::I64 { dims: db, data: b })] =>
            {
                let (dims, data) = zip_consts((&da, &a), (&db, &b), |x: i64, y: i64| x * y)?;
                Ok(Value::Const(Const::I64 { dims, data }))
            }
            [Value::Act(a), Value::Act(b)] => {
                let silu_of = |sig: usize, x: usize| matches!(self.acts[sig], Act::Sigmoid { input } if input == x);
                let input = if silu_of(b, a) {
                    a
                } else if silu_of(a, b) {
                    b
                } else {
                    return err("Mul of two activations that is not x * Sigmoid(x)");
                };
                Ok(Value::Act(self.push(Act::Silu { input })))
            }
            [Value::Act(a), Value::Const(Const::F32 { dims, data })] => {
                let Act::GnRestored {
                    input,
                    groups,
                    epsilon,
                } = self.acts[a].clone()
                else {
                    return err("Mul by a constant outside a GroupNorm affine");
                };
                let channels = is_channel_vector(&dims)
                    .ok_or_else(|| invalid_data(format!("GroupNorm gamma dims {dims:?}")))?;
                debug_assert_eq!(channels, data.len());
                Ok(Value::Act(self.push(Act::GnScaled {
                    input,
                    groups,
                    epsilon,
                    gamma: data,
                })))
            }
            _ => err("unsupported Mul operand kinds"),
        }
    }

    fn op_add(&mut self, node: &OnnxNode) -> io::Result<Value> {
        check_attributes(node, &[])?;
        match self.inputs::<2>(node)? {
            [Value::Act(a), Value::Act(b)] => {
                Ok(Value::Act(self.push(Act::Add { lhs: a, rhs: b })))
            }
            [Value::Act(a), Value::Const(Const::F32 { dims, data })] => {
                let Act::GnScaled {
                    input,
                    groups,
                    epsilon,
                    gamma,
                } = self.acts[a].clone()
                else {
                    return err("Add of a constant outside a GroupNorm affine");
                };
                let channels = is_channel_vector(&dims)
                    .ok_or_else(|| invalid_data(format!("GroupNorm beta dims {dims:?}")))?;
                if channels != gamma.len() {
                    return err(format!(
                        "GroupNorm beta has {channels} channels, gamma has {}",
                        gamma.len()
                    ));
                }
                if groups == 0 || channels % groups != 0 {
                    return err(format!("{groups} groups do not divide {channels} channels"));
                }
                Ok(Value::Act(self.push(Act::GroupNorm {
                    input,
                    norm: NormSpec {
                        channels,
                        groups,
                        epsilon,
                        gamma,
                        beta: data,
                    },
                })))
            }
            _ => err("unsupported Add operand kinds"),
        }
    }

    fn op_equal(&mut self, node: &OnnxNode) -> io::Result<Value> {
        check_attributes(node, &[])?;
        match self.inputs::<2>(node)? {
            [Value::Const(Const::I64 { dims: da, data: a }), Value::Const(Const::I64 { dims: db, data: b })] =>
            {
                let (dims, data) = zip_consts((&da, &a), (&db, &b), |x: i64, y: i64| x == y)?;
                Ok(Value::Const(Const::Bool { dims, data }))
            }
            _ => err("Equal is only supported on constant int64 tensors"),
        }
    }

    fn op_where(&mut self, node: &OnnxNode) -> io::Result<Value> {
        check_attributes(node, &[])?;
        let [Value::Const(Const::Bool {
            dims: dc,
            data: cond,
        }), Value::Const(Const::I64 { dims: dx, data: x }), Value::Const(Const::I64 { dims: dy, data: y })] =
            self.inputs::<3>(node)?
        else {
            return err("Where is only supported on constant (bool, int64, int64) tensors");
        };
        let (dims_xy, xy) = zip_consts((&dx, &x), (&dy, &y), |a: i64, b: i64| (a, b))?;
        let (dims, data) = zip_consts(
            (&dc, &cond),
            (&dims_xy, &xy),
            |c: bool, (a, b)| {
                if c {
                    a
                } else {
                    b
                }
            },
        )?;
        Ok(Value::Const(Const::I64 { dims, data }))
    }

    fn op_expand(&mut self, node: &OnnxNode) -> io::Result<Value> {
        check_attributes(node, &[])?;
        let [Value::Const(data), Value::Const(Const::I64 {
            dims: sd,
            data: shape,
        })] = self.inputs::<2>(node)?
        else {
            return err("Expand is only supported on constants");
        };
        if sd.len() != 1 {
            return err("Expand shape must be 1-D");
        }
        if broadcast_dims(data.dims(), &shape)? != data.dims() {
            return err(format!(
                "Expand of {:?} to {shape:?} would materialize new data",
                data.dims()
            ));
        }
        Ok(Value::Const(data))
    }

    fn op_concat(&mut self, node: &OnnxNode) -> io::Result<Value> {
        check_attributes(node, &["axis"])?;
        if int_attr(node, "axis")? != Some(1) {
            return err("Concat must be along axis 1");
        }
        let plane = [1, 1, BOARD_HEIGHT as i64, BOARD_WIDTH as i64];
        let [Value::Act(board), Value::Const(Const::F32 {
            dims: dr,
            data: row,
        }), Value::Const(Const::F32 {
            dims: dc,
            data: col,
        })] = self.inputs::<3>(node)?
        else {
            return err("Concat must be (board, row_coords, col_coords)");
        };
        if !matches!(self.acts[board], Act::Board) || dr != plane || dc != plane {
            return err("Concat must be (board, row_coords, col_coords) with [1,1,H,W] planes");
        }
        Ok(Value::Act(self.push(Act::BoardWithCoords { row, col })))
    }

    fn op_conv(&mut self, node: &OnnxNode) -> io::Result<Value> {
        check_attributes(
            node,
            &[
                "auto_pad",
                "dilations",
                "group",
                "kernel_shape",
                "pads",
                "strides",
            ],
        )?;
        let [Value::Act(input), Value::Const(Const::F32 {
            dims: wd,
            data: weight,
        }), Value::Const(Const::F32 {
            dims: bd,
            data: bias,
        })] = self.inputs::<3>(node)?
        else {
            return err("Conv must be (activation, constant weight, constant bias)");
        };
        let [out_c, in_c, 3, 3] = wd.as_slice() else {
            return err(format!("Conv weight dims {wd:?} are not [O, I, 3, 3]"));
        };
        if bd != [*out_c] || *out_c <= 0 || *in_c <= 0 {
            return err(format!("Conv bias dims {bd:?} do not match weight {wd:?}"));
        }
        match node.attribute("auto_pad") {
            None => {}
            Some(OnnxAttributeValue::String(mode)) if mode.as_slice() == b"NOTSET" => {}
            Some(other) => return err(format!("unsupported auto_pad {other:?}")),
        }
        if !matches!(
            ints_attr(node, "dilations")?.as_deref(),
            None | Some([1, 1])
        ) {
            return err("Conv dilations must be 1");
        }
        if !matches!(int_attr(node, "group")?, None | Some(1)) {
            return err("grouped Conv is not supported");
        }
        if !matches!(
            ints_attr(node, "kernel_shape")?.as_deref(),
            None | Some([3, 3])
        ) {
            return err("Conv kernel_shape must be [3, 3]");
        }
        if ints_attr(node, "pads")?.as_deref() != Some(&[1, 1, 1, 1][..]) {
            return err("Conv pads must be [1, 1, 1, 1]");
        }
        let stride = match ints_attr(node, "strides")?.as_deref() {
            None | Some([1, 1]) => 1,
            Some([2, 2]) => 2,
            Some(other) => return err(format!("unsupported Conv strides {other:?}")),
        };
        Ok(Value::Act(self.push(Act::Conv {
            input,
            spec: ConvSpec {
                in_channels: *in_c as usize,
                out_channels: *out_c as usize,
                stride,
                weight,
                bias,
            },
        })))
    }

    fn op_reshape(&mut self, node: &OnnxNode) -> io::Result<Value> {
        check_attributes(node, &["allowzero"])?;
        if !matches!(int_attr(node, "allowzero")?, None | Some(0)) {
            return err("Reshape allowzero must be 0");
        }
        match self.inputs::<2>(node)? {
            [Value::Act(input), Value::Const(Const::I64 { data: shape, .. })] => {
                match shape.as_slice() {
                    [0, groups, -1] if *groups > 0 => Ok(Value::Act(self.push(Act::GnGrouped {
                        input,
                        groups: *groups as usize,
                    }))),
                    [1, -1] => Ok(Value::Act(self.push(Act::Flatten { input }))),
                    other => err(format!("unsupported Reshape target {other:?}")),
                }
            }
            [Value::Act(normalized), Value::ShapeOf(original)] => {
                let Act::GnInstanceNorm {
                    input: grouped,
                    groups,
                    epsilon,
                } = self.acts[normalized].clone()
                else {
                    return err("Reshape-to-shape outside a GroupNorm decomposition");
                };
                let Act::GnGrouped { input: source, .. } = self.acts[grouped] else {
                    return err("GroupNorm InstanceNormalization input is not grouped");
                };
                if source != original {
                    return err("GroupNorm restores the shape of a different tensor");
                }
                Ok(Value::Act(self.push(Act::GnRestored {
                    input: source,
                    groups,
                    epsilon,
                })))
            }
            _ => err("unsupported Reshape operand kinds"),
        }
    }

    fn op_instance_norm(&mut self, node: &OnnxNode) -> io::Result<Value> {
        check_attributes(node, &["epsilon"])?;
        let epsilon = match node.attribute("epsilon") {
            None => 1e-5,
            Some(OnnxAttributeValue::Float(value)) => *value,
            Some(other) => return err(format!("invalid epsilon {other:?}")),
        };
        if !(epsilon.is_finite() && epsilon > 0.0 && epsilon < 1e-2) {
            return err(format!("implausible GroupNorm epsilon {epsilon}"));
        }
        let [Value::Act(grouped), Value::Const(scale), Value::Const(bias)] =
            self.inputs::<3>(node)?
        else {
            return err("InstanceNormalization must be (activation, constant, constant)");
        };
        let Act::GnGrouped { groups, .. } = self.acts[grouped] else {
            return err("InstanceNormalization input is not a GroupNorm grouping reshape");
        };
        let is_filled = |c: &Const, want: f32| matches!(c, Const::F32 { dims, data } if dims == &[groups as i64] && data.iter().all(|&v| v == want));
        if !is_filled(&scale, 1.0) || !is_filled(&bias, 0.0) || scale.len() != groups {
            return err("GroupNorm InstanceNormalization must use unit scale and zero bias");
        }
        Ok(Value::Act(self.push(Act::GnInstanceNorm {
            input: grouped,
            groups,
            epsilon,
        })))
    }
}

fn match_norm_silu(acts: &[Act], id: usize) -> io::Result<(NormSpec, usize)> {
    let Act::Silu { input } = &acts[id] else {
        return err("expected SiLU");
    };
    let Act::GroupNorm { input: inner, norm } = &acts[*input] else {
        return err("expected GroupNorm before SiLU");
    };
    Ok((norm.clone(), *inner))
}

fn match_conv(acts: &[Act], id: usize, stride: usize) -> io::Result<(ConvSpec, usize)> {
    match &acts[id] {
        Act::Conv { input, spec } if spec.stride == stride => Ok((spec.clone(), *input)),
        Act::Conv { spec, .. } => err(format!(
            "expected stride-{stride} Conv, found stride {}",
            spec.stride
        )),
        _ => err(format!("expected Conv, found {:?}", act_kind(&acts[id]))),
    }
}

fn act_kind(act: &Act) -> &'static str {
    match act {
        Act::Board => "Board",
        Act::BoardWithCoords { .. } => "BoardWithCoords",
        Act::Conv { .. } => "Conv",
        Act::GnGrouped { .. } => "GnGrouped",
        Act::GnInstanceNorm { .. } => "GnInstanceNorm",
        Act::GnRestored { .. } => "GnRestored",
        Act::GnScaled { .. } => "GnScaled",
        Act::GroupNorm { .. } => "GroupNorm",
        Act::Sigmoid { .. } => "Sigmoid",
        Act::Silu { .. } => "Silu",
        Act::Add { .. } => "Add",
        Act::Flatten { .. } => "Flatten",
    }
}

fn match_trunk(acts: &[Act], output: usize) -> io::Result<ConvTrunkSpec> {
    let Act::Flatten { input } = &acts[output] else {
        return err("graph output must be a flatten Reshape");
    };
    let (reduce_norm, reduce_id) = match_norm_silu(acts, *input)?;
    let (reduce, mut x) = match_conv(acts, reduce_id, 2)?;

    let mut blocks = Vec::new();
    while let Act::Add { lhs, rhs } = &acts[x] {
        let (skip, branch) = match (&acts[*lhs], &acts[*rhs]) {
            (_, Act::Conv { .. }) => (*lhs, *rhs),
            (Act::Conv { .. }, _) => (*rhs, *lhs),
            _ => return err("residual Add without a Conv branch"),
        };
        let (conv2, conv2_in) = match_conv(acts, branch, 1)?;
        let (norm2, conv1_id) = match_norm_silu(acts, conv2_in)?;
        let (conv1, conv1_in) = match_conv(acts, conv1_id, 1)?;
        let (norm1, block_in) = match_norm_silu(acts, conv1_in)?;
        if block_in != skip {
            return err("residual block normalizes a different tensor than its skip path");
        }
        blocks.push(ResidualBlockSpec {
            norm1,
            conv1,
            norm2,
            conv2,
        });
        x = skip;
    }
    blocks.reverse();

    let (stem_norm, stem_id) = match_norm_silu(acts, x)?;
    let (stem, stem_in) = match_conv(acts, stem_id, 1)?;
    let Act::BoardWithCoords { row, col } = &acts[stem_in] else {
        return err("stem Conv must consume concat(board, coords)");
    };

    let spec = ConvTrunkSpec {
        row_coords: row.clone(),
        col_coords: col.clone(),
        stem,
        stem_norm,
        blocks,
        reduce,
        reduce_norm,
    };
    validate_channels(&spec)?;
    Ok(spec)
}

fn validate_channels(spec: &ConvTrunkSpec) -> io::Result<()> {
    let c = spec.stem.out_channels;
    let cr = spec.reduce.out_channels;
    if spec.stem.in_channels != 3 {
        return err(format!(
            "stem Conv must have 3 input channels, got {}",
            spec.stem.in_channels
        ));
    }
    for (label, channels) in [("trunk", c), ("reduction", cr)] {
        if channels % CONV_NARROW_CHANNEL_TILE != 0 || channels > MAX_NORM_CHANNELS {
            return err(format!(
                "{label} channels {channels} must be a multiple of {CONV_NARROW_CHANNEL_TILE} and <= {MAX_NORM_CHANNELS}"
            ));
        }
    }
    let mut norms = vec![("stem", &spec.stem_norm, c)];
    let mut convs = vec![];
    for (i, block) in spec.blocks.iter().enumerate() {
        norms.push(("block norm1", &block.norm1, c));
        norms.push(("block norm2", &block.norm2, c));
        convs.push((format!("block {i} conv1"), &block.conv1));
        convs.push((format!("block {i} conv2"), &block.conv2));
    }
    norms.push(("reduction", &spec.reduce_norm, cr));
    for (label, norm, channels) in norms {
        if norm.channels != channels {
            return err(format!(
                "{label} GroupNorm has {} channels, expected {channels}",
                norm.channels
            ));
        }
    }
    for (label, conv) in convs {
        if conv.in_channels != c || conv.out_channels != c {
            return err(format!(
                "{label} must map {c} -> {c} channels, got {} -> {}",
                conv.in_channels, conv.out_channels
            ));
        }
    }
    if spec.reduce.in_channels != c {
        return err(format!(
            "reduction Conv input {} != trunk channels {c}",
            spec.reduce.in_channels
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Packed trunk + forward
// ---------------------------------------------------------------------------------------------

struct PackedBlock {
    norm1: GroupNormParams,
    conv1: PackedConv3x3,
    norm2: GroupNormParams,
    conv2: PackedConv3x3,
}

pub(crate) struct ConvTrunk {
    isa: Isa,
    channels: usize,
    reduce_channels: usize,
    /// Stem bias plus the constant coordinate-channel contributions, `[pixel][channel]`.
    stem_bias_map: Vec<f32>,
    /// Board-channel stem weights, `[tap][channel]`.
    stem_board_taps: Vec<f32>,
    stem_norm: GroupNormParams,
    blocks: Vec<PackedBlock>,
    reduce: PackedConv3x3,
    reduce_norm: GroupNormParams,
    full_table: ConvPixelTable,
    reduced_table: ConvPixelTable,
}

/// Per-instance activation buffers (one per `TetrisNN`, reused across calls).
pub(crate) struct ConvScratch {
    padded: Vec<f32>,
    x: Vec<f32>,
    h: Vec<f32>,
    reduced: Vec<f32>,
}

fn norm_params(spec: &NormSpec) -> GroupNormParams {
    GroupNormParams {
        channels: spec.channels,
        groups: spec.groups,
        epsilon: spec.epsilon,
        gamma: spec.gamma.clone(),
        beta: spec.beta.clone(),
    }
}

fn pack_conv(spec: &ConvSpec) -> PackedConv3x3 {
    PackedConv3x3::pack(
        &spec.weight,
        &spec.bias,
        spec.in_channels,
        spec.out_channels,
    )
}

fn pixel_table(
    out_height: usize,
    out_width: usize,
    stride: usize,
    in_channels: usize,
    block: usize,
) -> ConvPixelTable {
    let mut input_offsets = Vec::new();
    let mut output_rows = Vec::new();
    for oy in 0..out_height {
        for ox in 0..out_width {
            input_offsets.push((oy * stride * PADDED_WIDTH + ox * stride) * in_channels);
            output_rows.push(oy * out_width + ox);
        }
    }
    let dummy_row = output_rows.len();
    while input_offsets.len() % block != 0 {
        input_offsets.push(0);
        output_rows.push(dummy_row);
    }
    ConvPixelTable {
        input_offsets,
        output_rows,
    }
}

/// Interior of the zero-padded `(H+2) x (W+2)` activation buffer.
fn padded_interior(channels: usize) -> ImageLayout {
    ImageLayout {
        height: BOARD_HEIGHT,
        width: BOARD_WIDTH,
        offset: (PADDED_WIDTH + 1) * channels,
        row_pitch: PADDED_WIDTH,
    }
}

impl ConvTrunk {
    pub(crate) fn load(path: &Path) -> io::Result<Self> {
        Ok(Self::from_spec(
            &ConvTrunkSpec::from_onnx_path(path)?,
            Isa::detect(),
        ))
    }

    pub(crate) fn from_spec(spec: &ConvTrunkSpec, isa: Isa) -> Self {
        let c = spec.channels();
        let stem = &spec.stem;
        let tap_weight = |co: usize, ci: usize, tap: usize| stem.weight[(co * 3 + ci) * 9 + tap];

        let mut stem_bias_map = vec![0.0f32; PIXELS * c];
        for y in 0..BOARD_HEIGHT {
            for x in 0..BOARD_WIDTH {
                for co in 0..c {
                    let mut acc = f64::from(stem.bias[co]);
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let (iy, ix) = (y + ky, x + kx);
                            if iy == 0 || ix == 0 || iy > BOARD_HEIGHT || ix > BOARD_WIDTH {
                                continue;
                            }
                            let cell = (iy - 1) * BOARD_WIDTH + (ix - 1);
                            let tap = ky * 3 + kx;
                            acc += f64::from(spec.row_coords[cell])
                                * f64::from(tap_weight(co, 1, tap))
                                + f64::from(spec.col_coords[cell])
                                    * f64::from(tap_weight(co, 2, tap));
                        }
                    }
                    stem_bias_map[(y * BOARD_WIDTH + x) * c + co] = acc as f32;
                }
            }
        }
        let mut stem_board_taps = vec![0.0f32; 9 * c];
        for tap in 0..9 {
            for co in 0..c {
                stem_board_taps[tap * c + co] = tap_weight(co, 0, tap);
            }
        }

        let block_pixels = isa.conv_block_pixels();
        Self {
            isa,
            channels: c,
            reduce_channels: spec.reduce.out_channels,
            stem_bias_map,
            stem_board_taps,
            stem_norm: norm_params(&spec.stem_norm),
            blocks: spec
                .blocks
                .iter()
                .map(|block| PackedBlock {
                    norm1: norm_params(&block.norm1),
                    conv1: pack_conv(&block.conv1),
                    norm2: norm_params(&block.norm2),
                    conv2: pack_conv(&block.conv2),
                })
                .collect(),
            reduce: pack_conv(&spec.reduce),
            reduce_norm: norm_params(&spec.reduce_norm),
            full_table: pixel_table(BOARD_HEIGHT, BOARD_WIDTH, 1, c, block_pixels),
            reduced_table: pixel_table(REDUCED_HEIGHT, REDUCED_WIDTH, 2, c, block_pixels),
        }
    }

    pub(crate) fn isa(&self) -> Isa {
        self.isa
    }

    pub(crate) fn output_size(&self) -> usize {
        self.reduce_channels * REDUCED_PIXELS
    }

    /// PyTorch flattened-NCHW index of channels-last output feature `k`.
    pub(crate) fn feature_source_index(&self, k: usize) -> usize {
        let (pixel, channel) = (k / self.reduce_channels, k % self.reduce_channels);
        channel * REDUCED_PIXELS + pixel
    }

    pub(crate) fn new_scratch(&self) -> ConvScratch {
        let block = self.isa.conv_block_pixels();
        ConvScratch {
            padded: vec![0.0; PADDED_HEIGHT * PADDED_WIDTH * self.channels],
            x: vec![0.0; (PIXELS + block) * self.channels],
            h: vec![0.0; (PIXELS + block) * self.channels],
            reduced: vec![0.0; (REDUCED_PIXELS + block) * self.reduce_channels],
        }
    }

    /// Channels-last conv features for one board (`board` is the 200-cell input plane).
    ///
    /// # Safety
    /// `S` must be supported by the running CPU (callers dispatch on [`ConvTrunk::isa`]).
    #[inline(always)]
    pub(crate) unsafe fn forward_with<S: ConvKernel>(
        &self,
        board: &[f32],
        scratch: &mut ConvScratch,
        out: &mut [f32],
    ) {
        assert_eq!(board.len(), PIXELS, "board plane must have {PIXELS} cells");
        assert_eq!(out.len(), self.output_size());
        let c = self.channels;
        let ConvScratch {
            padded,
            x,
            h,
            reduced,
        } = scratch;

        self.stem_into::<S>(board, x);
        let dense = ImageLayout::dense(BOARD_HEIGHT, BOARD_WIDTH);
        group_norm_silu::<S>(
            x.as_ptr(),
            x.len(),
            x.as_mut_ptr(),
            x.len(),
            dense,
            &self.stem_norm,
        );

        let interior = padded_interior(c);
        for block in &self.blocks {
            group_norm_silu::<S>(
                x.as_ptr(),
                x.len(),
                padded.as_mut_ptr(),
                padded.len(),
                interior,
                &block.norm1,
            );
            S::conv3x3(
                padded,
                PADDED_WIDTH,
                &self.full_table,
                &block.conv1,
                h,
                false,
            );
            group_norm_silu::<S>(
                h.as_ptr(),
                h.len(),
                padded.as_mut_ptr(),
                padded.len(),
                interior,
                &block.norm2,
            );
            S::conv3x3(
                padded,
                PADDED_WIDTH,
                &self.full_table,
                &block.conv2,
                x,
                true,
            );
        }

        copy_image(&x[..PIXELS * c], padded, c, interior);
        S::conv3x3(
            padded,
            PADDED_WIDTH,
            &self.reduced_table,
            &self.reduce,
            reduced,
            false,
        );
        group_norm_silu::<S>(
            reduced.as_ptr(),
            reduced.len(),
            out.as_mut_ptr(),
            out.len(),
            ImageLayout::dense(REDUCED_HEIGHT, REDUCED_WIDTH),
            &self.reduce_norm,
        );
    }

    /// `x = stem bias map + board-channel contributions` (sparse over non-zero cells).
    #[inline(always)]
    unsafe fn stem_into<S: Simd>(&self, board: &[f32], x: &mut [f32]) {
        let c = self.channels;
        x[..PIXELS * c].copy_from_slice(&self.stem_bias_map);
        for (cell, &value) in board.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            let (row, col) = (cell / BOARD_WIDTH, cell % BOARD_WIDTH);
            let v = S::splat(value);
            for ky in 0..3 {
                // Output pixel (y, x) reads input (y + ky - 1, x + kx - 1).
                let Some(y) = (row + 1).checked_sub(ky).filter(|&y| y < BOARD_HEIGHT) else {
                    continue;
                };
                for kx in 0..3 {
                    let Some(xx) = (col + 1).checked_sub(kx).filter(|&xx| xx < BOARD_WIDTH) else {
                        continue;
                    };
                    let dst = x.as_mut_ptr().add((y * BOARD_WIDTH + xx) * c);
                    let taps = self.stem_board_taps.as_ptr().add((ky * 3 + kx) * c);
                    for c0 in (0..c).step_by(S::LANES) {
                        let acc = S::load(dst.add(c0));
                        v.mul_add(S::load(taps.add(c0)), acc).store(dst.add(c0));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
