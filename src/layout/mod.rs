use crate::types::{Graph, LayoutAlgorithm};

pub mod traits;
pub mod algorithms;

pub use traits::*;
pub use algorithms::*;

/// Common trait for all layout algorithms
pub trait LayoutEngine {
    /// Apply the layout algorithm to a graph
    fn apply_layout(&self, graph: &mut Graph) -> Result<(), String>;
    
    /// Get the name of the layout algorithm
    fn name(&self) -> &'static str;
    
    /// Get a description of the layout algorithm
    fn description(&self) -> &'static str;
}

/// Apply a layout algorithm to a graph
pub fn apply_layout(graph: &mut Graph, layout: &LayoutAlgorithm) -> Result<(), String> {
    match layout {
        // fCoSE layout (compound)
        // fCoSE layout (constraint)
        // Circle layout
        // AVSDF layout
        // Grid layout
        // CoSE layout
        // CoSE Bilkent layout
        // CoSE Bilkent layout (compound)
        // Cola layout
        // Cola layout (compound)
        // Euler layout
        // Spread layout
        LayoutAlgorithm::Fcose(options) => algorithms::fcose::apply_layout(graph, options),
        LayoutAlgorithm::CoseBilkent(options) => algorithms::cose_bilkent::apply_layout(graph, options),
        LayoutAlgorithm::Cise(options) => algorithms::cise::apply_layout(graph, options),
        LayoutAlgorithm::Concentric(options) => algorithms::concentric::apply_layout(graph, options),
        LayoutAlgorithm::KlayLayered(options) => algorithms::klay::apply_layout(graph, options),
        LayoutAlgorithm::Dagre(options) => algorithms::dagre::apply_layout(graph, options),
    }
}