//! Minimal, dependency-free reader for the parts of an ONNX `ModelProto` that the native
//! heads implementation needs: graph initializers (f32 weights), nodes (op type, inputs and
//! scalar attributes) and graph input/output names.
//!
//! Only the protobuf wire format is decoded here; validating that the graph has the expected
//! structure is the caller's job (see `heads.rs`).

use std::collections::HashMap;
use std::fmt;

/// ONNX `TensorProto.DataType.FLOAT`.
const ONNX_DATA_TYPE_FLOAT: u64 = 1;
/// ONNX `TensorProto.DataLocation.EXTERNAL`.
const ONNX_DATA_LOCATION_EXTERNAL: u64 = 1;

#[derive(Debug)]
pub(super) struct OnnxParseError(String);

impl fmt::Display for OnnxParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ONNX parse error: {}", self.0)
    }
}

impl std::error::Error for OnnxParseError {}

type ParseResult<T> = Result<T, OnnxParseError>;

fn parse_error<T>(message: impl Into<String>) -> ParseResult<T> {
    Err(OnnxParseError(message.into()))
}

pub(super) struct OnnxTensor {
    pub(super) dims: Vec<usize>,
    pub(super) data: Vec<f32>,
}

#[derive(Default)]
pub(super) struct OnnxAttribute {
    pub(super) float: Option<f32>,
    pub(super) int: Option<i64>,
}

pub(super) struct OnnxNode {
    pub(super) op_type: String,
    pub(super) inputs: Vec<String>,
    pub(super) outputs: Vec<String>,
    pub(super) attributes: HashMap<String, OnnxAttribute>,
}

pub(super) struct OnnxGraph {
    pub(super) nodes: Vec<OnnxNode>,
    pub(super) initializers: HashMap<String, OnnxTensor>,
    pub(super) input_names: Vec<String>,
    pub(super) output_names: Vec<String>,
}

enum WireValue<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

struct WireReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> WireReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read_varint(&mut self) -> ParseResult<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let Some(&byte) = self.buf.get(self.pos) else {
                return parse_error("truncated varint");
            };
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        parse_error("varint longer than 10 bytes")
    }

    fn take(&mut self, len: usize) -> ParseResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.buf.len())
            .ok_or_else(|| OnnxParseError("length-delimited field overruns buffer".into()))?;
        let bytes = &self.buf[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    /// Next `(field_number, value)` pair, or `None` at the end of the message.
    fn next_field(&mut self) -> ParseResult<Option<(u64, WireValue<'a>)>> {
        if self.pos == self.buf.len() {
            return Ok(None);
        }
        let key = self.read_varint()?;
        let field = key >> 3;
        let value = match key & 0x7 {
            0 => WireValue::Varint(self.read_varint()?),
            1 => {
                self.take(8)?;
                WireValue::Fixed64
            }
            2 => {
                let len = usize::try_from(self.read_varint()?)
                    .map_err(|_| OnnxParseError("length does not fit in usize".into()))?;
                WireValue::Bytes(self.take(len)?)
            }
            5 => {
                let bytes = self.take(4)?;
                WireValue::Fixed32(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            }
            wire_type => return parse_error(format!("unsupported wire type {wire_type}")),
        };
        Ok(Some((field, value)))
    }
}

fn utf8(bytes: &[u8], what: &str) -> ParseResult<String> {
    String::from_utf8(bytes.to_vec()).map_err(|_| OnnxParseError(format!("{what} is not UTF-8")))
}

fn expect_bytes<'a>(value: WireValue<'a>, what: &str) -> ParseResult<&'a [u8]> {
    match value {
        WireValue::Bytes(bytes) => Ok(bytes),
        _ => parse_error(format!("{what}: expected a length-delimited field")),
    }
}

fn expect_varint(value: WireValue<'_>, what: &str) -> ParseResult<u64> {
    match value {
        WireValue::Varint(v) => Ok(v),
        _ => parse_error(format!("{what}: expected a varint field")),
    }
}

/// Repeated varint field that may be encoded packed or one element per field.
fn push_varints(value: WireValue<'_>, out: &mut Vec<u64>, what: &str) -> ParseResult<()> {
    match value {
        WireValue::Varint(v) => out.push(v),
        WireValue::Bytes(bytes) => {
            let mut reader = WireReader::new(bytes);
            while reader.pos < bytes.len() {
                out.push(reader.read_varint()?);
            }
        }
        _ => return parse_error(format!("{what}: expected varints")),
    }
    Ok(())
}

/// Repeated float field that may be encoded packed or one element per field.
fn push_floats(value: WireValue<'_>, out: &mut Vec<f32>, what: &str) -> ParseResult<()> {
    match value {
        WireValue::Fixed32(bits) => out.push(f32::from_bits(bits)),
        WireValue::Bytes(bytes) => {
            if bytes.len() % 4 != 0 {
                return parse_error(format!("{what}: packed float length not a multiple of 4"));
            }
            out.extend(
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
        }
        _ => return parse_error(format!("{what}: expected floats")),
    }
    Ok(())
}

fn parse_tensor(bytes: &[u8]) -> ParseResult<(String, OnnxTensor)> {
    let mut name = String::new();
    let mut dims = Vec::new();
    let mut data_type = 0u64;
    let mut float_data = Vec::new();
    let mut raw_data: &[u8] = &[];
    let mut data_location = 0u64;

    let mut reader = WireReader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => push_varints(value, &mut dims, "TensorProto.dims")?,
            2 => data_type = expect_varint(value, "TensorProto.data_type")?,
            4 => push_floats(value, &mut float_data, "TensorProto.float_data")?,
            8 => name = utf8(expect_bytes(value, "TensorProto.name")?, "tensor name")?,
            9 => raw_data = expect_bytes(value, "TensorProto.raw_data")?,
            14 => data_location = expect_varint(value, "TensorProto.data_location")?,
            _ => {}
        }
    }

    if data_location == ONNX_DATA_LOCATION_EXTERNAL {
        return parse_error(format!(
            "initializer '{name}' uses external data (unsupported)"
        ));
    }
    if data_type != ONNX_DATA_TYPE_FLOAT {
        return parse_error(format!(
            "initializer '{name}' has data type {data_type}, expected FLOAT"
        ));
    }
    let dims = dims
        .into_iter()
        .map(|d| usize::try_from(d).map_err(|_| OnnxParseError(format!("bad dim in '{name}'"))))
        .collect::<ParseResult<Vec<usize>>>()?;
    let data = if raw_data.is_empty() {
        float_data
    } else {
        if raw_data.len() % 4 != 0 {
            return parse_error(format!(
                "initializer '{name}' raw_data length not a multiple of 4"
            ));
        }
        raw_data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };
    let expected_len: usize = dims.iter().product();
    if data.len() != expected_len {
        return parse_error(format!(
            "initializer '{name}' has {} values but dims {:?}",
            data.len(),
            dims
        ));
    }
    Ok((name, OnnxTensor { dims, data }))
}

fn parse_attribute(bytes: &[u8]) -> ParseResult<(String, OnnxAttribute)> {
    let mut name = String::new();
    let mut attribute = OnnxAttribute::default();
    let mut reader = WireReader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => {
                name = utf8(
                    expect_bytes(value, "AttributeProto.name")?,
                    "attribute name",
                )?
            }
            2 => match value {
                WireValue::Fixed32(bits) => attribute.float = Some(f32::from_bits(bits)),
                _ => return parse_error("AttributeProto.f: expected fixed32"),
            },
            // int64 attributes are two's-complement varints.
            3 => attribute.int = Some(expect_varint(value, "AttributeProto.i")? as i64),
            _ => {}
        }
    }
    Ok((name, attribute))
}

fn parse_node(bytes: &[u8]) -> ParseResult<OnnxNode> {
    let mut node = OnnxNode {
        op_type: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attributes: HashMap::new(),
    };
    let mut reader = WireReader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => node
                .inputs
                .push(utf8(expect_bytes(value, "NodeProto.input")?, "node input")?),
            2 => node.outputs.push(utf8(
                expect_bytes(value, "NodeProto.output")?,
                "node output",
            )?),
            4 => node.op_type = utf8(expect_bytes(value, "NodeProto.op_type")?, "op type")?,
            5 => {
                let (name, attribute) =
                    parse_attribute(expect_bytes(value, "NodeProto.attribute")?)?;
                node.attributes.insert(name, attribute);
            }
            _ => {}
        }
    }
    Ok(node)
}

fn parse_value_info_name(bytes: &[u8]) -> ParseResult<String> {
    let mut reader = WireReader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        if field == 1 {
            return utf8(expect_bytes(value, "ValueInfoProto.name")?, "value name");
        }
    }
    parse_error("ValueInfoProto without a name")
}

fn parse_graph(bytes: &[u8]) -> ParseResult<OnnxGraph> {
    let mut graph = OnnxGraph {
        nodes: Vec::new(),
        initializers: HashMap::new(),
        input_names: Vec::new(),
        output_names: Vec::new(),
    };
    let mut reader = WireReader::new(bytes);
    while let Some((field, value)) = reader.next_field()? {
        match field {
            1 => graph
                .nodes
                .push(parse_node(expect_bytes(value, "GraphProto.node")?)?),
            5 => {
                let (name, tensor) = parse_tensor(expect_bytes(value, "GraphProto.initializer")?)?;
                if graph.initializers.insert(name.clone(), tensor).is_some() {
                    return parse_error(format!("duplicate initializer '{name}'"));
                }
            }
            11 => graph.input_names.push(parse_value_info_name(expect_bytes(
                value,
                "GraphProto.input",
            )?)?),
            12 => graph.output_names.push(parse_value_info_name(expect_bytes(
                value,
                "GraphProto.output",
            )?)?),
            _ => {}
        }
    }
    // Older exporters list initializers among the graph inputs; keep only real inputs.
    graph
        .input_names
        .retain(|name| !graph.initializers.contains_key(name));
    Ok(graph)
}

/// Decode the main graph of a serialized ONNX `ModelProto`.
pub(super) fn parse_model_graph(bytes: &[u8]) -> ParseResult<OnnxGraph> {
    let mut reader = WireReader::new(bytes);
    let mut graph = None;
    while let Some((field, value)) = reader.next_field()? {
        if field == 7 {
            graph = Some(parse_graph(expect_bytes(value, "ModelProto.graph")?)?);
        }
    }
    graph.ok_or_else(|| OnnxParseError("ModelProto has no graph".into()))
}

/// Protobuf encoding helpers for building ONNX graphs in tests.
#[cfg(test)]
pub(super) mod proto_writer {
    pub(in crate::inference) fn varint(mut value: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    pub(in crate::inference) fn bytes_field(field: u64, payload: &[u8], out: &mut Vec<u8>) {
        varint(field << 3 | 2, out);
        varint(payload.len() as u64, out);
        out.extend_from_slice(payload);
    }

    pub(in crate::inference) fn varint_field(field: u64, value: u64, out: &mut Vec<u8>) {
        varint(field << 3, out);
        varint(value, out);
    }
}

#[cfg(test)]
mod tests {
    use super::proto_writer::{bytes_field, varint, varint_field};
    use super::*;

    #[test]
    fn parses_initializers_nodes_and_io_names() {
        let mut tensor = Vec::new();
        varint_field(1, 2, &mut tensor); // dims: unpacked
        varint_field(1, 3, &mut tensor);
        varint_field(2, ONNX_DATA_TYPE_FLOAT, &mut tensor);
        bytes_field(8, b"layer.weight", &mut tensor);
        let raw: Vec<u8> = (0..6)
            .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
            .collect();
        bytes_field(9, &raw, &mut tensor);

        let mut packed_tensor = Vec::new();
        bytes_field(1, &[2], &mut packed_tensor); // dims: packed
        varint_field(2, ONNX_DATA_TYPE_FLOAT, &mut packed_tensor);
        bytes_field(8, b"layer.bias", &mut packed_tensor);
        let floats: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        bytes_field(4, &floats, &mut packed_tensor);

        let mut attribute = Vec::new();
        bytes_field(1, b"epsilon", &mut attribute);
        varint(2 << 3 | 5, &mut attribute);
        attribute.extend_from_slice(&1e-5f32.to_le_bytes());
        let mut int_attribute = Vec::new();
        bytes_field(1, b"axis", &mut int_attribute);
        varint_field(3, (-1i64) as u64, &mut int_attribute);

        let mut node = Vec::new();
        bytes_field(1, b"x", &mut node);
        bytes_field(1, b"layer.weight", &mut node);
        bytes_field(2, b"y", &mut node);
        bytes_field(4, b"LayerNormalization", &mut node);
        bytes_field(5, &attribute, &mut node);
        bytes_field(5, &int_attribute, &mut node);

        let mut input = Vec::new();
        bytes_field(1, b"x", &mut input);
        let mut output = Vec::new();
        bytes_field(1, b"y", &mut output);

        let mut graph = Vec::new();
        bytes_field(1, &node, &mut graph);
        bytes_field(5, &tensor, &mut graph);
        bytes_field(5, &packed_tensor, &mut graph);
        bytes_field(11, &input, &mut graph);
        bytes_field(12, &output, &mut graph);

        let mut model = Vec::new();
        varint_field(1, 8, &mut model); // ir_version, ignored
        bytes_field(7, &graph, &mut model);

        let parsed = parse_model_graph(&model).expect("parse");
        assert_eq!(parsed.input_names, vec!["x".to_string()]);
        assert_eq!(parsed.output_names, vec!["y".to_string()]);
        let weight = &parsed.initializers["layer.weight"];
        assert_eq!(weight.dims, vec![2, 3]);
        assert_eq!(weight.data, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        let bias = &parsed.initializers["layer.bias"];
        assert_eq!(bias.dims, vec![2]);
        assert_eq!(bias.data, vec![1.5, -2.0]);
        assert_eq!(parsed.nodes.len(), 1);
        let node = &parsed.nodes[0];
        assert_eq!(node.op_type, "LayerNormalization");
        assert_eq!(
            node.inputs,
            vec!["x".to_string(), "layer.weight".to_string()]
        );
        assert_eq!(node.attributes["epsilon"].float, Some(1e-5));
        assert_eq!(node.attributes["axis"].int, Some(-1));
    }

    #[test]
    fn rejects_truncated_input() {
        let mut graph = Vec::new();
        bytes_field(1, b"abc", &mut graph);
        let mut model = Vec::new();
        bytes_field(7, &graph, &mut model);
        model.truncate(model.len() - 1);
        assert!(parse_model_graph(&model).is_err());
    }
}
