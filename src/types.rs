use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for nodes and edges
pub type Id = String;

/// Key-value pair for metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetadataValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

/// Node in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: Id,
    pub position: Option<(f64, f64)>,
    pub metadata: HashMap<String, MetadataValue>,
}

impl Node {
    pub fn new(id: impl Into<Id>) -> Self {
        Self {
            id: id.into(),
            position: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.position = Some((x, y));
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Edge in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: Id,
    pub source: Id,
    pub target: Id,
    pub metadata: HashMap<String, MetadataValue>,
}

impl Edge {
    pub fn new(id: impl Into<Id>, source: impl Into<Id>, target: impl Into<Id>) -> Self {
        Self {
            id: id.into(),
            source: source.into(),
            target: target.into(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Complete graph structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Graph {
    pub nodes: HashMap<Id, Node>,
    pub edges: HashMap<Id, Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) -> &mut Self {
        self.nodes.insert(node.id.clone(), node);
        self
    }

    pub fn add_edge(&mut self, edge: Edge) -> &mut Self {
        self.edges.insert(edge.id.clone(), edge);
        self
    }

    pub fn remove_node(&mut self, id: &Id) -> Option<Node> {
        // Also remove any edges connected to this node
        let edges_to_remove: Vec<Id> = self.edges.values()
            .filter(|e| e.source == *id || e.target == *id)
            .map(|e| e.id.clone())
            .collect();
        
        for edge_id in edges_to_remove {
            self.edges.remove(&edge_id);
        }
        
        self.nodes.remove(id)
    }

    pub fn remove_edge(&mut self, id: &Id) -> Option<Edge> {
        self.edges.remove(id)
    }
}

/// Base layout configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutOptions {
    pub padding: u32,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            padding: 30,
        }
    }
}

// Implement From traits for MetadataValue
impl From<String> for MetadataValue {
    fn from(value: String) -> Self {
        MetadataValue::String(value)
    }
}

impl From<&str> for MetadataValue {
    fn from(value: &str) -> Self {
        MetadataValue::String(value.to_string())
    }
}

impl From<f64> for MetadataValue {
    fn from(value: f64) -> Self {
        MetadataValue::Number(value)
    }
}

impl From<i32> for MetadataValue {
    fn from(value: i32) -> Self {
        MetadataValue::Number(value as f64)
    }
}

impl From<bool> for MetadataValue {
    fn from(value: bool) -> Self {
        MetadataValue::Boolean(value)
    }
}
