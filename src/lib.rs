use wasm_bindgen::prelude::*;

mod types;
mod layout;

use layout::LayoutEngine;
pub use types::{Graph, Node, Edge, Id, MetadataValue, LayoutOptions};
pub use layout::algorithms::fcose::{FcoseLayoutEngine, FcoseOptions};

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function to get better error messages if our code ever panics.
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct LayoutManager {
    graph: Graph,
}

#[wasm_bindgen]
impl LayoutManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        set_panic_hook();
        Self {
            graph: Graph::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, id: String, x: Option<f64>, y: Option<f64>) {
        let mut node = Node::new(id);
        if let (Some(x_val), Some(y_val)) = (x, y) {
            node = node.with_position(x_val, y_val);
        }
        self.graph.add_node(node);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, id: String, source: String, target: String) {
        let edge = Edge::new(id, source, target);
        self.graph.add_edge(edge);
    }

    /// Remove a node from the graph
    pub fn remove_node(&mut self, id: String) {
        self.graph.remove_node(&id);
    }

    /// Remove an edge from the graph
    pub fn remove_edge(&mut self, id: String) {
        self.graph.remove_edge(&id);
    }

    /// Apply the fCoSE layout algorithm
    pub fn apply_fcose_layout(&mut self, options_json: String) -> Result<String, JsValue> {
        let options: FcoseOptions = serde_json::from_str(&options_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse options: {}", e)))?;
        
        FcoseLayoutEngine::new(options)
            .apply_layout(&mut self.graph)
            .map_err(|e| JsValue::from_str(&format!("Layout error: {}", e)))?;
        
        // Return the updated graph as JSON
        serde_json::to_string(&self.graph)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize graph: {}", e)))
    }

    /// Get the current graph state as JSON
    pub fn get_graph_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.graph)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize graph: {}", e)))
    }

    /// Load a graph from JSON
    pub fn load_graph_json(&mut self, json: String) -> Result<(), JsValue> {
        self.graph = serde_json::from_str(&json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse graph: {}", e)))?;
        Ok(())
    }
}
