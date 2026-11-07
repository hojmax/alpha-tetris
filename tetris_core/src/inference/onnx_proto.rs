//! Minimal, dependency-free decoder for the subset of the ONNX protobuf schema needed to read
//! exported weight graphs: graph nodes (with attributes), initializers, and graph inputs/outputs.
//!
//! Anything outside that subset that would change the meaning of the graph (external tensor
//! data, sparse initializers, model-local functions, unsupported tensor element types) is
//! rejected instead of silently ignored. Purely informational fields (names, doc strings,
//! producer metadata, value_info) are skipped.

use std::io;
use std::path::Path;

pub(crate) fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[derive(Clone, Debug)]
pub(crate) struct OnnxModel {
    /// Opset version of the default (`ai.onnx`) domain.
    pub opset_version: i64,
    pub graph: OnnxGraph,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OnnxGraph {
    pub nodes: Vec<OnnxNode>,
    pub initializers: Vec<OnnxTensor>,
    pub inputs: Vec<OnnxValueInfo>,
    pub outputs: Vec<OnnxValueInfo>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OnnxNode {
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<OnnxAttribute>,
}

#[derive(Clone, Debug)]
pub(crate) struct OnnxAttribute {
    pub name: String,
    pub value: OnnxAttributeValue,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OnnxAttributeValue {
    Float(f32),
    Int(i64),
    String(Vec<u8>),
    Tensor(OnnxTensor),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    /// Attribute kinds the loaders never need (graphs, sparse tensors, ...); keeps the type id.
    Unsupported(i64),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OnnxTensor {
    pub name: String,
    pub dims: Vec<i64>,
    pub data: OnnxTensorData,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OnnxTensorData {
    Float(Vec<f32>),
    Int64(Vec<i64>),
}

#[derive(Clone, Debug, Default)]
pub(crate) struct OnnxValueInfo {
    pub name: String,
    pub elem_type: i64,
    pub dims: Vec<OnnxDim>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OnnxDim {
    Value(i64),
    Param(String),
    Unknown,
}

pub(crate) const ONNX_ELEM_FLOAT: i64 = 1;
pub(crate) const ONNX_ELEM_INT64: i64 = 7;

const ATTR_FLOAT: i64 = 1;
const ATTR_INT: i64 = 2;
const ATTR_STRING: i64 = 3;
const ATTR_TENSOR: i64 = 4;
const ATTR_FLOATS: i64 = 6;
const ATTR_INTS: i64 = 7;

impl OnnxNode {
    pub(crate) fn attribute(&self, name: &str) -> Option<&OnnxAttributeValue> {
        self.attributes
            .iter()
            .find(|attr| attr.name == name)
            .map(|attr| &attr.value)
    }
}

impl OnnxTensor {
    pub(crate) fn num_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }
}

pub(crate) fn load_model(path: &Path) -> io::Result<OnnxModel> {
    let bytes = std::fs::read(path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to read ONNX model {}: {e}", path.display()),
        )
    })?;
    decode_model(&bytes).map_err(|e| invalid_data(format!("{}: {e}", path.display())))
}

pub(crate) fn decode_model(bytes: &[u8]) -> io::Result<OnnxModel> {
    let mut graph = None;
    let mut opset_version = None;
    let mut reader = Reader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            7 => graph = Some(decode_graph(value.bytes("ModelProto.graph")?)?),
            8 => {
                let (domain, version) = decode_opset(value.bytes("ModelProto.opset_import")?)?;
                if domain.is_empty() || domain == "ai.onnx" {
                    opset_version = Some(version);
                }
            }
            25 => return Err(invalid_data("model-local ONNX functions are not supported")),
            _ => {}
        }
    }
    Ok(OnnxModel {
        opset_version: opset_version
            .ok_or_else(|| invalid_data("ONNX model has no default-domain opset import"))?,
        graph: graph.ok_or_else(|| invalid_data("ONNX model has no graph"))?,
    })
}

fn decode_opset(bytes: &[u8]) -> io::Result<(String, i64)> {
    let mut domain = String::new();
    let mut version = 0;
    let mut reader = Reader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => domain = value.string("OperatorSetIdProto.domain")?,
            2 => version = value.int64("OperatorSetIdProto.version")?,
            _ => {}
        }
    }
    Ok((domain, version))
}

fn decode_graph(bytes: &[u8]) -> io::Result<OnnxGraph> {
    let mut graph = OnnxGraph::default();
    let mut reader = Reader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => graph
                .nodes
                .push(decode_node(value.bytes("GraphProto.node")?)?),
            5 => graph
                .initializers
                .push(decode_tensor(value.bytes("GraphProto.initializer")?)?),
            11 => graph
                .inputs
                .push(decode_value_info(value.bytes("GraphProto.input")?)?),
            12 => graph
                .outputs
                .push(decode_value_info(value.bytes("GraphProto.output")?)?),
            15 => return Err(invalid_data("sparse initializers are not supported")),
            _ => {}
        }
    }
    Ok(graph)
}

fn decode_node(bytes: &[u8]) -> io::Result<OnnxNode> {
    let mut node = OnnxNode::default();
    let mut reader = Reader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => node.inputs.push(value.string("NodeProto.input")?),
            2 => node.outputs.push(value.string("NodeProto.output")?),
            4 => node.op_type = value.string("NodeProto.op_type")?,
            5 => node
                .attributes
                .push(decode_attribute(value.bytes("NodeProto.attribute")?)?),
            7 => node.domain = value.string("NodeProto.domain")?,
            _ => {}
        }
    }
    if node.op_type.is_empty() {
        return Err(invalid_data("ONNX node without op_type"));
    }
    Ok(node)
}

fn decode_attribute(bytes: &[u8]) -> io::Result<OnnxAttribute> {
    let mut name = String::new();
    let mut attr_type = None;
    let mut float = 0.0f32;
    let mut int = 0i64;
    let mut string = Vec::new();
    let mut tensor = None;
    let mut floats = Vec::new();
    let mut ints = Vec::new();
    let mut reader = Reader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => name = value.string("AttributeProto.name")?,
            2 => float = value.float32("AttributeProto.f")?,
            3 => int = value.int64("AttributeProto.i")?,
            4 => string = value.bytes("AttributeProto.s")?.to_vec(),
            5 => tensor = Some(decode_tensor(value.bytes("AttributeProto.t")?)?),
            7 => value.extend_float32s(&mut floats, "AttributeProto.floats")?,
            8 => value.extend_int64s(&mut ints, "AttributeProto.ints")?,
            20 => attr_type = Some(value.int64("AttributeProto.type")?),
            _ => {}
        }
    }
    let attr_type =
        attr_type.ok_or_else(|| invalid_data(format!("attribute '{name}' has no type")))?;
    let value = match attr_type {
        ATTR_FLOAT => OnnxAttributeValue::Float(float),
        ATTR_INT => OnnxAttributeValue::Int(int),
        ATTR_STRING => OnnxAttributeValue::String(string),
        ATTR_TENSOR => OnnxAttributeValue::Tensor(
            tensor.ok_or_else(|| invalid_data(format!("tensor attribute '{name}' is empty")))?,
        ),
        ATTR_FLOATS => OnnxAttributeValue::Floats(floats),
        ATTR_INTS => OnnxAttributeValue::Ints(ints),
        other => OnnxAttributeValue::Unsupported(other),
    };
    Ok(OnnxAttribute { name, value })
}

fn decode_tensor(bytes: &[u8]) -> io::Result<OnnxTensor> {
    let mut name = String::new();
    let mut dims = Vec::new();
    let mut data_type = 0i64;
    let mut raw_data: Option<&[u8]> = None;
    let mut float_data = Vec::new();
    let mut int64_data = Vec::new();
    let mut reader = Reader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => value.extend_int64s(&mut dims, "TensorProto.dims")?,
            2 => data_type = value.int64("TensorProto.data_type")?,
            3 => return Err(invalid_data("segmented tensors are not supported")),
            4 => value.extend_float32s(&mut float_data, "TensorProto.float_data")?,
            5 | 6 | 10 | 11 => {
                return Err(invalid_data(format!(
                    "tensor uses unsupported typed data field {field}"
                )))
            }
            7 => value.extend_int64s(&mut int64_data, "TensorProto.int64_data")?,
            8 => name = value.string("TensorProto.name")?,
            9 => raw_data = Some(value.bytes("TensorProto.raw_data")?),
            13 | 14 => {
                return Err(invalid_data(format!(
                    "tensor '{name}' uses external data, which is not supported"
                )))
            }
            _ => {}
        }
    }
    if dims.iter().any(|&d| d < 0) {
        return Err(invalid_data(format!(
            "tensor '{name}' has negative dims {dims:?}"
        )));
    }
    let expected: usize = dims.iter().map(|&d| d as usize).product();
    let data = match data_type {
        ONNX_ELEM_FLOAT => OnnxTensorData::Float(match raw_data {
            Some(raw) => {
                if !float_data.is_empty() {
                    return Err(invalid_data(format!(
                        "tensor '{name}' has both raw_data and float_data"
                    )));
                }
                if raw.len() % 4 != 0 {
                    return Err(invalid_data(format!(
                        "tensor '{name}' raw_data length {} is not a multiple of 4",
                        raw.len()
                    )));
                }
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            None => float_data,
        }),
        ONNX_ELEM_INT64 => OnnxTensorData::Int64(match raw_data {
            Some(raw) => {
                if !int64_data.is_empty() {
                    return Err(invalid_data(format!(
                        "tensor '{name}' has both raw_data and int64_data"
                    )));
                }
                if raw.len() % 8 != 0 {
                    return Err(invalid_data(format!(
                        "tensor '{name}' raw_data length {} is not a multiple of 8",
                        raw.len()
                    )));
                }
                raw.chunks_exact(8)
                    .map(|b| i64::from_le_bytes(b.try_into().expect("chunk of 8 bytes")))
                    .collect()
            }
            None => int64_data,
        }),
        other => {
            return Err(invalid_data(format!(
                "tensor '{name}' has unsupported element type {other}"
            )))
        }
    };
    let actual = match &data {
        OnnxTensorData::Float(values) => values.len(),
        OnnxTensorData::Int64(values) => values.len(),
    };
    if actual != expected {
        return Err(invalid_data(format!(
            "tensor '{name}' has {actual} elements but dims {dims:?} imply {expected}"
        )));
    }
    Ok(OnnxTensor { name, dims, data })
}

fn decode_value_info(bytes: &[u8]) -> io::Result<OnnxValueInfo> {
    let mut info = OnnxValueInfo::default();
    let mut reader = Reader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => info.name = value.string("ValueInfoProto.name")?,
            2 => decode_type(value.bytes("ValueInfoProto.type")?, &mut info)?,
            _ => {}
        }
    }
    Ok(info)
}

fn decode_type(bytes: &[u8], info: &mut OnnxValueInfo) -> io::Result<()> {
    let mut reader = Reader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => {}
            4 | 5 | 8 | 9 => {
                return Err(invalid_data(format!(
                    "value '{}' is not a plain tensor type",
                    info.name
                )))
            }
            _ => continue,
        }
        let mut tensor_reader = Reader::new(value.bytes("TypeProto.tensor_type")?);
        while let Some((tensor_field, tensor_value)) = tensor_reader.next_field()? {
            match tensor_field {
                1 => info.elem_type = tensor_value.int64("TypeProto.Tensor.elem_type")?,
                2 => info.dims = decode_shape(tensor_value.bytes("TypeProto.Tensor.shape")?)?,
                _ => {}
            }
        }
    }
    Ok(())
}

fn decode_shape(bytes: &[u8]) -> io::Result<Vec<OnnxDim>> {
    let mut dims = Vec::new();
    let mut reader = Reader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        if field != 1 {
            continue;
        }
        let mut dim = OnnxDim::Unknown;
        let mut dim_reader = Reader::new(value.bytes("TensorShapeProto.dim")?);
        while let Some((dim_field, dim_value)) = dim_reader.next_field()? {
            match dim_field {
                1 => dim = OnnxDim::Value(dim_value.int64("Dimension.dim_value")?),
                2 => dim = OnnxDim::Param(dim_value.string("Dimension.dim_param")?),
                _ => {}
            }
        }
        dims.push(dim);
    }
    Ok(dims)
}

#[derive(Clone, Copy)]
enum WireValue<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> WireValue<'a> {
    fn bytes(self, what: &str) -> io::Result<&'a [u8]> {
        match self {
            WireValue::Bytes(bytes) => Ok(bytes),
            _ => Err(invalid_data(format!(
                "{what}: expected length-delimited field"
            ))),
        }
    }

    fn string(self, what: &str) -> io::Result<String> {
        String::from_utf8(self.bytes(what)?.to_vec())
            .map_err(|_| invalid_data(format!("{what}: invalid UTF-8")))
    }

    fn int64(self, what: &str) -> io::Result<i64> {
        match self {
            WireValue::Varint(value) => Ok(value as i64),
            _ => Err(invalid_data(format!("{what}: expected varint field"))),
        }
    }

    fn float32(self, what: &str) -> io::Result<f32> {
        match self {
            WireValue::Fixed32(bits) => Ok(f32::from_bits(bits)),
            _ => Err(invalid_data(format!("{what}: expected 32-bit field"))),
        }
    }

    /// Repeated int64: accepts both packed and unpacked encodings.
    fn extend_int64s(self, out: &mut Vec<i64>, what: &str) -> io::Result<()> {
        match self {
            WireValue::Varint(value) => out.push(value as i64),
            WireValue::Bytes(bytes) => {
                let mut reader = Reader::new(bytes);
                while !reader.is_empty() {
                    out.push(reader.varint()? as i64);
                }
            }
            _ => return Err(invalid_data(format!("{what}: expected varint data"))),
        }
        Ok(())
    }

    /// Repeated float: accepts both packed and unpacked encodings.
    fn extend_float32s(self, out: &mut Vec<f32>, what: &str) -> io::Result<()> {
        match self {
            WireValue::Fixed32(bits) => out.push(f32::from_bits(bits)),
            WireValue::Bytes(bytes) => {
                if bytes.len() % 4 != 0 {
                    return Err(invalid_data(format!("{what}: packed floats length")));
                }
                out.extend(
                    bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                );
            }
            _ => return Err(invalid_data(format!("{what}: expected float data"))),
        }
        Ok(())
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> io::Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| invalid_data("truncated protobuf varint"))?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(invalid_data("protobuf varint longer than 10 bytes"))
    }

    fn take(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.buf.len())
            .ok_or_else(|| invalid_data("truncated protobuf field"))?;
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn next_field(&mut self) -> io::Result<Option<(u32, WireValue<'a>)>> {
        if self.is_empty() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = (key >> 3) as u32;
        if field == 0 {
            return Err(invalid_data("protobuf field number 0"));
        }
        let value = match key & 7 {
            0 => WireValue::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                WireValue::Fixed64
            }
            2 => {
                let len = usize::try_from(self.varint()?)
                    .map_err(|_| invalid_data("protobuf length overflow"))?;
                WireValue::Bytes(self.take(len)?)
            }
            5 => WireValue::Fixed32(u32::from_le_bytes(
                self.take(4)?.try_into().expect("4 bytes"),
            )),
            wire => {
                return Err(invalid_data(format!(
                    "unsupported protobuf wire type {wire} (field {field})"
                )))
            }
        };
        Ok(Some((field, value)))
    }
}
