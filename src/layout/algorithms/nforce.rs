// oc_layout.rs

//! Multi-level Force-Directed Placement (FDP) graph layout algorithm (Custom Graph).
//!
//! Implements a layout algorithm based on the research outlined in the associated
//! design document. It combines a multi-level approach (graph coarsening/uncoarsening)
//! inspired by Walshaw with physics-based refinement using force-directed placement
//! accelerated by an N-dimensional spatial tree (N-tree), aiming for O(N log N) complexity.
//!
//! This implementation uses a custom graph representation (SimpleGraph) instead of petgraph
//! and targets a generic `Layout` trait interface compatible with this representation.

use nalgebra::{Point, VectorN};
use std::marker::PhantomData;
use std::collections::HashMap;

// --- Placeholder Dependencies (Replace with actual crates) ---

// N-Tree implementation (e.g., based on zhifeng_impl_barnes_hut_tree recommendation)
// Assuming a trait or struct like `BarnesHutTree` exists
mod n_tree {
    use nalgebra::{Point, VectorN};

    // Placeholder - requires an actual N-Tree implementation
    pub trait NTree<const D: usize> {
        // Builds the tree from node positions and their masses/weights.
        fn build(points: &[Point<f64, D>], weights: &[f64]) -> Self;
        // Computes the approximate repulsive force on a node `node_idx` at `pos`.
        fn compute_force(
            &self,
            node_idx: usize,      // Index of the node we are calculating force FOR
            pos: &Point<f64, D>,  // Position of the node
            theta: f64,           // Barnes-Hut approximation parameter
            k_squared: f64,       // Ideal distance squared (k_l * k_l)
            repulsive_constant: f64, // Scaling constant C
        ) -> VectorN<f64, D>;
    }

    // Dummy implementation for compilation
    #[derive(Debug)]
    pub struct DummyNTree;
    impl<const D: usize> NTree<D> for DummyNTree {
        fn build(_points: &[Point<f64, D>], _weights: &[f64]) -> Self { DummyNTree }
        fn compute_force(
            &self,
            _node_idx: usize,
            _pos: &Point<f64, D>,
            _theta: f64,
            _k_squared: f64,
            _repulsive_constant: f64,
        ) -> VectorN<f64, D> {
            VectorN::zeros()
        }
    }
}
use n_tree::{NTree, DummyNTree}; // Use the actual NTree implementation

// Optimization library (e.g., argmin - optional but recommended)
// Assuming configuration structs for optimizers exist
mod optimizer {
    use nalgebra::{Point, VectorN};
    use std::time::Instant; // For basic timing/logging

    // Placeholder - requires an actual optimizer setup (like argmin)
    #[derive(Clone, Debug)]
    pub enum OptimizerConfig {
        AdaptiveGradientDescent {
            initial_temp: f64,
            cooling_factor: f64,
            tolerance: f64, // Based on max displacement relative to temperature/k
            max_iterations: usize,
        },
        // Add other optimizers like SimulatedAnnealing if needed
    }

    // Dummy function for compilation - Represents the optimization loop
    // Takes the current layout and a function to calculate forces/displacements
    pub fn run_optimization<const D: usize>(
        config: &OptimizerConfig,
        initial_layout: Vec<Point<f64, D>>,
        // Closure takes current positions, returns total displacements for this step
        force_and_displacement_calc: impl Fn(&[Point<f64, D>], f64) -> (Vec<VectorN<f64, D>>, f64),
        initial_k_l: f64, // Initial ideal distance for this refinement level
        node_count: usize,
    ) -> Vec<Point<f64, D>> {
        if node_count == 0 { return initial_layout; }

        let mut current_layout = initial_layout;
        match config {
            OptimizerConfig::AdaptiveGradientDescent {
                initial_temp,
                cooling_factor,
                tolerance,
                max_iterations
            } => {
                let mut temperature = *initial_temp;
                let min_temp_threshold = tolerance * initial_k_l * 0.1; // Stop if temp gets tiny

                for iter in 0..*max_iterations {
                    let start_time = Instant::now();

                    // 1. Calculate forces/displacements based on current layout
                    // The closure encapsulates the N-Tree build + force calculation logic
                    let (displacements, max_displacement_sq) = force_and_displacement_calc(&current_layout, temperature);
                    let max_displacement = max_displacement_sq.sqrt();


                    // 2. Update positions (apply displacements limited by temperature)
                    for i in 0..node_count {
                       // Note: displacements already capped inside force_calc usually
                       current_layout[i] += displacements[i];
                    }

                    // 3. Cooling
                    temperature *= cooling_factor;

                    let duration = start_time.elapsed();
                     println!(
                         "    Refine Iter {}: MaxDisp={:.4e}, Temp={:.4e}, Time={:?}",
                         iter + 1, max_displacement, temperature, duration
                     );

                    // 4. Check convergence
                    // Stop if max displacement is small relative to ideal length * tolerance
                    // Or if temperature is very low
                    if max_displacement < (*tolerance * initial_k_l) || temperature < min_temp_threshold {
                         println!("    Convergence reached at iteration {}.", iter + 1);
                        break;
                    }
                     if iter == *max_iterations - 1 {
                         println!("    Max iterations ({}) reached.", *max_iterations);
                     }
                }
            } // Add other optimizer logic here
        }
        current_layout
    }
}
use optimizer::{OptimizerConfig, run_optimization};
use rand::seq::SliceRandom; // For shuffling node indices
use rand::Rng; // For random placement and perturbation

// --- Custom Graph Representation (No Petgraph) ---

/// Represents data associated with a node in the graph.
/// `N` is the user-provided node data type.
#[derive(Clone, Debug, Default)]
pub struct NodeData<N> {
    /// User-defined data associated with the node.
    pub user_data: N,
    // --- Internal data used by the layout algorithm ---
    /// Aggregated weight/mass, used in coarsening and Barnes-Hut.
    pub(crate) mass: f64,
    /// Tracks original node indices during coarsening.
    pub(crate) original_indices: Vec<usize>,
}

/// Represents data associated with an edge in the graph.
/// `E` is the user-provided edge data type.
#[derive(Clone, Debug, Default)]
pub struct EdgeData<E> {
    /// User-defined data associated with the edge.
    pub user_data: E,
    // --- Internal data used by the layout algorithm ---
    /// Aggregated weight/strength, used in coarsening and potentially forces.
    pub(crate) weight: f64,
}

/// A simple graph representation using an adjacency list.
/// Does not use `petgraph`. Indices are simple `usize`.
/// `N` is the user-provided node data type.
/// `E` is the user-provided edge data type.
#[derive(Clone, Debug, Default)]
pub struct SimpleGraph<N, E> {
    /// Stores data for each node. The index in this vector is the node ID.
    pub nodes: Vec<NodeData<N>>,
    /// Adjacency list representation. `adj[i]` contains a list of tuples,
    /// where each tuple is `(neighbor_index, edge_data)`.
    /// Assumes undirected edges are stored twice (once for each direction).
    pub adj: Vec<Vec<(usize, EdgeData<E>)>>,
}

impl<N: Default, E: Default> SimpleGraph<N, E> {
    /// Creates a new empty graph.
    pub fn new() -> Self {
        SimpleGraph {
            nodes: Vec::new(),
            adj: Vec::new(),
        }
    }

    /// Adds a new node with default user data and returns its index.
    pub fn add_node(&mut self, user_data: N) -> usize {
        let index = self.nodes.len();
        self.nodes.push(NodeData {
            user_data,
            mass: 1.0, // Default initial mass
            original_indices: vec![index], // Initially, it represents itself
        });
        self.adj.push(Vec::new()); // Add adjacency list for the new node
        index
    }

    /// Adds an undirected edge between two nodes.
    /// Stores the edge in both nodes' adjacency lists.
    pub fn add_edge(&mut self, u: usize, v: usize, user_data: E) {
         // Ensure nodes exist
        assert!(u < self.nodes.len(), "Node index {} out of bounds.", u);
        assert!(v < self.nodes.len(), "Node index {} out of bounds.", v);
        if u == v { return; } // Ignore self-loops for typical layout

        let edge_data = EdgeData {
            user_data,
            weight: 1.0, // Default initial weight
        };

        // Check for duplicate edges before adding
        if !self.adj[u].iter().any(|(neighbor, _)| *neighbor == v) {
            self.adj[u].push((v, edge_data.clone()));
        }
        if !self.adj[v].iter().any(|(neighbor, _)| *neighbor == u) {
            self.adj[v].push((u, edge_data));
        }
    }

    /// Returns the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges in the graph (counting each undirected edge once).
    pub fn edge_count(&self) -> usize {
        self.adj.iter().map(|neighbors| neighbors.len()).sum::<usize>() / 2
    }

     /// Provides an iterator over node indices (0 to n-1).
    pub fn node_indices(&self) -> impl Iterator<Item = usize> {
        0..self.nodes.len()
    }

     /// Provides an iterator over neighbors of a given node.
    pub fn neighbors(&self, u: usize) -> impl Iterator<Item = &(usize, EdgeData<E>)> {
         assert!(u < self.adj.len(), "Node index {} out of bounds.", u);
         self.adj[u].iter()
    }
}

// --- Layout Interface Definition (Custom Graph) ---

/// Generic trait for graph layout algorithms operating on `SimpleGraph`.
pub trait Layout<N, E, const D: usize> {
    /// Computes the node positions for a given graph.
    ///
    /// # Arguments
    /// * `graph`: A reference to the `SimpleGraph` to be laid out.
    ///
    /// # Returns
    /// * A `Vec<nalgebra::Point<f64, D>>` where the position at index `i`
    ///   corresponds to the node `graph.nodes[i]`.
    fn layout(&self, graph: &SimpleGraph<N, E>) -> Vec<Point<f64, D>>;
}

// --- Algorithm Configuration Structs (Same as before) ---

/// Strategy for graph coarsening (Section 4.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoarseningStrategy {
    /// Matches nodes based on the minimum weight neighbor (Walshaw's heuristic).
    WalshawSmallestWeight,
    // Add other strategies like HeavyEdgeMatching if needed
}

/// Force model for FDP refinement (Section 5.1).
#[derive(Clone, Debug)]
pub enum ForceModel {
    /// Walshaw's modified Fruchterman-Reingold forces.
    WalshawModifiedFR {
        /// Scaling constant for repulsive force (C). Typically 0.5-0.9.
        repulsive_constant: f64,
    },
    // Add other models like T_FDP if needed
}

/// Represents the mapping between graph levels during coarsening/uncoarsening.
/// Uses `usize` indices directly.
#[derive(Clone, Debug)]
struct LevelMapping {
    /// Maps each node index in the finer graph (G_l) to its corresponding node index
    /// in the coarser graph (G_{l+1}). `fine_to_coarse[fine_node_idx] = coarse_node_idx`.
    fine_to_coarse: Vec<usize>,
    /// Maps each node index in the coarser graph (G_{l+1}) to the list of node indices
    /// in the finer graph (G_l) that were merged into it.
    /// `coarse_to_fine[coarse_node_idx] = [fine_node_idx1, fine_node_idx2, ...]`.
    coarse_to_fine: HashMap<usize, Vec<usize>>,
    // Could potentially store aggregated weights or other level-specific info here too.
}

// Type aliases for internal graph structures used during coarsening/refinement
type InternalNodeData = (); // No specific internal node data needed beyond mass/orig_idx
type InternalEdgeData = (); // No specific internal edge data needed beyond weight
type InternalGraph = SimpleGraph<InternalNodeData, InternalEdgeData>;


// --- MultiLevelLayout Algorithm Implementation (Custom Graph) ---

/// Implements the multi-level, physics-based graph layout algorithm using `SimpleGraph`.
///
/// Generic Parameters:
/// * `UserDataN`: Original node weight type provided by the user.
/// * `UserDataE`: Original edge weight type provided by the user.
/// * `D`: Dimensionality of the layout (e.g., 2 for 2D, 3 for 3D).
pub struct MultiLevelLayout<UserDataN, UserDataE, const D: usize>
where
    UserDataN: Clone + Default + Send + Sync,
    UserDataE: Clone + Default + Send + Sync,
{
    // --- Coarsening Parameters (Section 4.1 & 6) ---
    pub coarsening_strategy: CoarseningStrategy,
    pub coarsening_threshold: usize,
    pub min_coarsening_ratio: f64,

    // --- Refinement Parameters (Section 5 & 6) ---
    pub force_model: ForceModel,
    pub barnes_hut_theta: f64,
    pub optimizer_config: OptimizerConfig,
    pub coarsest_layout_iterations: usize,

    _phantom: PhantomData<(UserDataN, UserDataE)>,
}

impl<UserDataN, UserDataE, const D: usize> MultiLevelLayout<UserDataN, UserDataE, D>
where
    UserDataN: Clone + Default + Send + Sync,
    UserDataE: Clone + Default + Send + Sync,
    // Add any bounds needed by N-Tree, Optimizer, Math operations
{
    /// Creates a new `MultiLevelLayout` instance with specified parameters.
    pub fn new(
        coarsening_strategy: CoarseningStrategy,
        coarsening_threshold: usize,
        min_coarsening_ratio: f64,
        force_model: ForceModel,
        barnes_hut_theta: f64,
        optimizer_config: OptimizerConfig,
        coarsest_layout_iterations: usize,
    ) -> Self {
        Self {
            coarsening_strategy,
            coarsening_threshold,
            min_coarsening_ratio,
            force_model,
            barnes_hut_theta,
            optimizer_config,
            coarsest_layout_iterations,
            _phantom: PhantomData,
        }
    }

    // --- Internal Helper Methods ---

    /// Performs the graph coarsening phase (Section 4.1).
    /// Creates a hierarchy of `InternalGraph`s and the mappings between them.
    ///
    /// **Note:** This implementation needs the actual matching and aggregation logic.
    fn coarsen_graph(
        &self,
        initial_graph: &InternalGraph, // G_0 with internal weights/mass set
    ) -> (Vec<InternalGraph>, Vec<LevelMapping>) {
        println!("Running coarsening phase...");
        let mut hierarchy = vec![initial_graph.clone()]; // Start with G_0
        let mut mappings = Vec::new();
        let mut rng = rand::thread_rng();

        loop {
            let current_graph = hierarchy.last().unwrap();
            let num_nodes = current_graph.node_count();

            if num_nodes <= self.coarsening_threshold {
                println!("Coarsening stopped: Node count ({}) <= threshold ({}).", num_nodes, self.coarsening_threshold);
                break;
            }

            println!("Coarsening level {} ({} nodes)...", mappings.len(), num_nodes);

            // --- Perform one level of coarsening (G_l -> G_{l+1}) ---
            // TODO: Implement actual matching algorithm (e.g., WalshawSmallestWeight)
            // 1. Create random permutation of node indices (0..num_nodes)
            let mut node_indices: Vec<usize> = (0..num_nodes).collect();
            node_indices.shuffle(&mut rng);

            // 2. Perform matching based on strategy (Placeholder: random pairing)
            let mut matched = vec![false; num_nodes];
            let mut fine_to_coarse_map = vec![usize::MAX; num_nodes]; // usize::MAX indicates not mapped yet
            let mut coarse_nodes_data = Vec::new();
            let mut coarse_to_fine_map = HashMap::new();

            for &fine_idx1 in &node_indices {
                if matched[fine_idx1] { continue; }

                // Placeholder: Just pair sequentially for now, skipping matched ones
                let mut fine_idx2 = None;
                 for &potential_match in &node_indices {
                     if fine_idx1 != potential_match && !matched[potential_match] {
                         // In a real implementation, check neighbors and weights here based on strategy
                         fine_idx2 = Some(potential_match);
                         break;
                     }
                 }


                let coarse_node_idx = coarse_nodes_data.len();
                let mut merged_mass = 0.0;
                let mut original_fine_indices = Vec::new();

                // Create coarse node by merging fine_idx1 and fine_idx2 (or just fine_idx1 if no partner)
                matched[fine_idx1] = true;
                fine_to_coarse_map[fine_idx1] = coarse_node_idx;
                merged_mass += current_graph.nodes[fine_idx1].mass;
                original_fine_indices.extend(current_graph.nodes[fine_idx1].original_indices.iter());


                if let Some(idx2) = fine_idx2 {
                    matched[idx2] = true;
                    fine_to_coarse_map[idx2] = coarse_node_idx;
                    merged_mass += current_graph.nodes[idx2].mass;
                     original_fine_indices.extend(current_graph.nodes[idx2].original_indices.iter());
                    coarse_to_fine_map.insert(coarse_node_idx, vec![fine_idx1, idx2]);
                } else {
                     // Matched with self (or couldn't find partner in simple placeholder)
                     coarse_to_fine_map.insert(coarse_node_idx, vec![fine_idx1]);
                }

                coarse_nodes_data.push(NodeData {
                    user_data: (), // Internal graphs don't need user data
                    mass: merged_mass,
                    original_indices: original_fine_indices, // Keep track of original nodes
                });
            }

             // Ensure all nodes were mapped (should be true if matching is correct)
             assert!(fine_to_coarse_map.iter().all(|&idx| idx != usize::MAX), "Not all fine nodes were mapped to coarse nodes!");

            // 3. Build the coarse graph structure (nodes + edges)
            let mut coarse_graph = InternalGraph {
                nodes: coarse_nodes_data,
                adj: vec![Vec::new(); coarse_nodes_data.len()],
            };

            // TODO: Aggregate edge weights correctly
            // Iterate through edges of the *fine* graph. Find corresponding coarse nodes.
            // Add edges to the coarse graph, summing weights if multiple fine edges map to the same coarse edge.
            // Use a temporary HashMap<(usize, usize), f64> to accumulate weights.
             println!("  -> Coarse graph has {} nodes.", coarse_graph.node_count());
             if coarse_graph.node_count() == num_nodes {
                  println!("Coarsening stalled (no reduction). Stopping.");
                  break; // Avoid infinite loops if coarsening doesn't reduce size
             }


            // Check stalling condition
            let ratio = coarse_graph.node_count() as f64 / num_nodes as f64;
            if ratio >= self.min_coarsening_ratio && num_nodes > self.coarsening_threshold {
                println!("Coarsening stopped: Ratio ({:.2}) >= min ratio ({:.2}).", ratio, self.min_coarsening_ratio);
                break;
            }

            let level_mapping = LevelMapping {
                fine_to_coarse: fine_to_coarse_map,
                coarse_to_fine: coarse_to_fine_map,
            };

            hierarchy.push(coarse_graph);
            mappings.push(level_mapping);

            if mappings.len() > 30 { // Safety break
                 println!("Warning: Max coarsening levels reached.");
                 break;
            }
        }

        println!("Coarsening finished. Hierarchy depth: {}", hierarchy.len());
        (hierarchy, mappings)
    }

    /// Computes the initial layout for the coarsest graph (G_L) (Section 4.2).
    fn compute_initial_layout(
        &self,
        coarsest_graph: &InternalGraph,
    ) -> Vec<Point<f64, D>> {
        println!("Computing initial layout for coarsest graph ({} nodes)...", coarsest_graph.node_count());
        let num_nodes = coarsest_graph.node_count();
        if num_nodes == 0 {
            return Vec::new();
        }

        // Initial Random Placement
        let mut rng = rand::thread_rng();
        let scale = (num_nodes as f64).sqrt() * 10.0; // Heuristic scaling
        let initial_layout: Vec<Point<f64, D>> = (0..num_nodes)
            .map(|_| {
                let mut coords = nalgebra::SVector::<f64, D>::zeros();
                for i in 0..D {
                    coords[i] = rng.gen::<f64>() * scale - scale / 2.0;
                }
                Point::from(coords)
            })
            .collect();

        // Refine the initial layout using specific iteration count
        let coarsest_optimizer_config = OptimizerConfig::AdaptiveGradientDescent {
             max_iterations: self.coarsest_layout_iterations,
            ..match self.optimizer_config {
                 OptimizerConfig::AdaptiveGradientDescent { initial_temp, cooling_factor, tolerance, .. } =>
                    OptimizerConfig::AdaptiveGradientDescent { initial_temp, cooling_factor, tolerance, max_iterations: 0},
             }
        };

        println!("Refining coarsest layout ({} iterations)...", self.coarsest_layout_iterations);
        let final_layout = self.refine_layout(
            coarsest_graph,
            initial_layout,
            &coarsest_optimizer_config,
            None, // No k_{l+1} for the coarsest level
        );

        final_layout
    }

    /// Interpolates the layout from a coarser level (G_{l+1}) to a finer level (G_l) (Section 4.3).
    fn interpolate_layout(
        &self,
        fine_graph: &InternalGraph, // G_l
        coarse_layout: &[Point<f64, D>], // Layout for G_{l+1}
        mapping: &LevelMapping,         // Mapping from G_l to G_{l+1}
        k_l: f64,                       // Ideal distance at finer level (for perturbation)
    ) -> Vec<Point<f64, D>> {
        println!("Interpolating layout to finer level ({} nodes)...", fine_graph.node_count());
        let num_fine_nodes = fine_graph.node_count();
        let mut fine_layout = vec![Point::<f64, D>::origin(); num_fine_nodes];
        let mut rng = rand::thread_rng();
        let perturbation_scale = k_l * 0.1; // Small random offset

        for fine_idx in 0..num_fine_nodes {
            let coarse_idx = mapping.fine_to_coarse[fine_idx];
            if coarse_idx >= coarse_layout.len() {
                // This can happen if fine_to_coarse wasn't fully populated or if indices are wrong
                eprintln!("Warning: Coarse index {} out of bounds for fine node {}. Using origin.", coarse_idx, fine_idx);
                fine_layout[fine_idx] = Point::origin(); // Fallback
                continue;
            }
            let base_pos = coarse_layout[coarse_idx];

            // Add small random perturbation
            let mut displacement = VectorN::<f64, D>::zeros();
            for i in 0..D {
                displacement[i] = (rng.gen::<f64>() - 0.5) * perturbation_scale;
            }
            fine_layout[fine_idx] = base_pos + displacement;
        }

        fine_layout
    }

    /// Performs FDP refinement on a given graph level (G_l) (Section 5).
    fn refine_layout(
        &self,
        graph: &InternalGraph,
        initial_layout: Vec<Point<f64, D>>,
        optimizer_config: &OptimizerConfig, // Use specific config for this level
        k_l_plus_1: Option<f64>, // Ideal distance from coarser level G_{l+1}
    ) -> Vec<Point<f64, D>> {
        let num_nodes = graph.node_count();
        if num_nodes <= 1 { // Cannot layout 0 or 1 node graph with forces
            return initial_layout;
        }
        println!("Refining layout for level with {} nodes...", num_nodes);

        // --- Calculate k_l (Ideal Distance for this level) --- Section 5.1
        let k_l = match k_l_plus_1 {
            // k_l = k_{l+1} / sqrt(2) according to Walshaw's paper (page 8, eq 4)
            Some(k_prev) => k_prev / (2.0f64.sqrt()),
            // Estimate k_L for the coarsest graph (if not passed)
            // Common heuristic: C * sqrt(Area / N) -> C' * L / sqrt(N)
            // Or simply based on average edge length of initial random layout (less robust)
            None => {
                // Using Area heuristic assuming layout area is roughly square
                 let layout_width = (num_nodes as f64).sqrt() * 10.0; // Consistent with random placement scale
                 1.0 * layout_width / (num_nodes as f64).sqrt() // Simplified C'=1.0
            }
        };
        let k_l_squared = k_l * k_l;
        println!("  Ideal distance k_l = {:.4e}", k_l);

        // --- Prepare data for force calculations ---
        let node_masses: Vec<f64> = graph.nodes.iter().map(|n| n.mass).collect();
        let (repulsive_constant C) = match self.force_model {
             ForceModel::WalshawModifiedFR { repulsive_constant } => (repulsive_constant),
             // Handle other force models if added
        };


        // --- Define the Force Calculation Logic (Closure for Optimizer) ---
        let force_and_displacement_calc = |current_layout: &[Point<f64, D>], temperature: f64| -> (Vec<VectorN<f64, D>>, f64) {

            // 1. Build the N-Tree (rebuilt each iteration)
             // Use DummyNTree for now; replace with actual NTree implementation
            let n_tree = DummyNTree::build(current_layout, &node_masses); // Replace DummyNTree

            let mut net_displacements = vec![VectorN::<f64, D>::zeros(); num_nodes];
            let mut max_displacement_sq = 0.0f64;

            // 2. Calculate Forces for each node u
            for u in 0..num_nodes {
                let pos_u = current_layout[u];
                let mut net_force_u = VectorN::<f64, D>::zeros();

                // a) Attractive forces (from neighbors v) - Section 5.1
                // Walshaw implies Fa = (d^2 / k_l) * unit_vector(v-u)
                 // FR typically Fa = (d / k) * (v-u) = (d^2 / k) * unit_vector * (1/d) -> Needs clarification
                 // Let's use FR standard: F_a = d^2 / k_l * unit_vector
                for (v_idx, edge_data) in graph.neighbors(u) {
                    let v = *v_idx;
                    if u == v { continue; } // Skip self-loops if any
                    let pos_v = current_layout[v];
                    let diff = pos_v - pos_u;
                    let dist_sq = diff.norm_squared();

                    if dist_sq > 1e-9 { // Avoid division by zero / instability
                        let dist = dist_sq.sqrt();
                        let attractive_force_magnitude = dist_sq / k_l; // FR formula F = x^2/k
                        let force_vec = diff * (attractive_force_magnitude / dist); // (diff / dist) * magnitude
                        net_force_u += force_vec; // Force pulling u towards v
                    }
                }

                // b) Repulsive forces (from all other nodes v, using N-Tree) - Section 5.2
                 let repulsive_force: VectorN<f64, D> = n_tree.compute_force(
                     u,
                     &pos_u,
                     self.barnes_hut_theta,
                     k_l_squared,
                     C,
                 );
                 // Walshaw's model seems to use F_r = - C * w_v * k_l^2 / d * unit_vector
                 // DummyNTree returns zero now, needs real implementation based on chosen N-tree crate.
                 // Note: BH force usually approximates SUM(- C * w_v * k_l^2 / d^2 * diff), check N-Tree impl.
                 net_force_u += repulsive_force; // Add repulsive force contribution


                // 3. Calculate Displacement and Apply Cooling/Temperature Limit
                let displacement = net_force_u; // Simplest: displacement proportional to force
                let displacement_norm_sq = displacement.norm_squared();

                 if displacement_norm_sq > 1e-9 {
                     let displacement_norm = displacement_norm_sq.sqrt();
                     // Limit displacement magnitude by temperature (Fruchterman-Reingold cooling)
                     let limited_displacement = displacement * (temperature.min(displacement_norm) / displacement_norm);
                     net_displacements[u] = limited_displacement;
                     max_displacement_sq = max_displacement_sq.max(limited_displacement.norm_squared());
                 } else {
                      net_displacements[u] = VectorN::zeros(); // No force, no displacement
                 }
            }
            (net_displacements, max_displacement_sq)
        };


        // --- Run the Optimization Loop ---
        let final_layout = run_optimization(
            optimizer_config,
            initial_layout,
            force_and_displacement_calc,
            k_l,
            num_nodes,
        );

        final_layout
    }
}

// --- Implement the Layout Trait ---

impl<UserDataN, UserDataE, const D: usize> Layout<UserDataN, UserDataE, D>
    for MultiLevelLayout<UserDataN, UserDataE, D>
where
    UserDataN: Clone + Default + Send + Sync,
    UserDataE: Clone + Default + Send + Sync,
    // Add bounds required by N-Tree, Optimizer, etc.
{
    /// Computes the graph layout using the multi-level V-cycle.
    fn layout(
        &self,
        graph: &SimpleGraph<UserDataN, UserDataE>,
    ) -> Vec<Point<f64, D>> {
        let start_time = std::time::Instant::now();
        let num_nodes = graph.node_count();
        println!(
            "Starting MultiLevelLayout for graph with {} nodes, {} edges.",
            num_nodes, graph.edge_count()
        );
         if num_nodes == 0 { return Vec::new(); }
         if num_nodes == 1 { return vec![Point::origin()]; } // Layout for single node

        // --- Step 1: Create G_0 with internal weights ---
         let mut initial_internal_graph = InternalGraph {
              nodes: Vec::with_capacity(num_nodes),
              adj: Vec::with_capacity(num_nodes),
         };
         for (idx, node_data) in graph.nodes.iter().enumerate() {
              initial_internal_graph.nodes.push(NodeData {
                   user_data: (), // Discard user data for internal graph
                   mass: 1.0,    // Initialize mass (can be customized later)
                   original_indices: vec![idx], // Track original index
              });
              initial_internal_graph.adj.push(Vec::new()); // Initialize adj list
         }
         // Copy adjacency structure
         for u in 0..num_nodes {
              for (v_idx, edge_data) in graph.neighbors(u) {
                   // Avoid adding duplicates if graph represents undirected edges symmetrically
                   if u < *v_idx {
                        let internal_edge_data = EdgeData {
                             user_data: (),
                             weight: 1.0, // Initialize weight (can be customized)
                        };
                        // Add edge symmetrically in internal graph
                         initial_internal_graph.adj[u].push((*v_idx, internal_edge_data.clone()));
                         initial_internal_graph.adj[*v_idx].push((u, internal_edge_data));
                   }
              }
         }


        // --- Step 2: Coarsening Phase ---
        let (hierarchy, mappings) = self.coarsen_graph(&initial_internal_graph);
        if hierarchy.is_empty() {
             eprintln!("Error: Coarsening produced an empty hierarchy.");
             // Fallback: Layout the original graph directly
              let fallback_layout = self.refine_layout(
                  &initial_internal_graph,
                  vec![Point::origin(); num_nodes], // Start from origin? Or random?
                  &self.optimizer_config,
                  None
              );
              return fallback_layout;
        }

        // --- Step 3: Initial Layout Phase (Coarsest Graph) ---
        let coarsest_graph = hierarchy.last().unwrap();
        let mut current_layout = self.compute_initial_layout(coarsest_graph);
        let mut current_k = None; // k_l will be calculated inside refine_layout for the first time

        // --- Step 4: Uncoarsening and Refinement Phase ---
        println!("Starting uncoarsening and refinement phase...");
        // Iterate from G_{L-1} down to G_0
        for level in (0..hierarchy.len() - 1).rev() {
            println!("--- Uncoarsening to Level {} ---", level);
            let fine_graph = &hierarchy[level];
            // Calculate k_l for the *fine* graph based on the *coarse* graph's k_{l+1}
             // The k calculated *inside* refine_layout for the coarse graph becomes k_{l+1} here
             // We need to capture the k_l calculated in the previous refine_layout call
            let k_l = match current_k {
                 Some(k_prev) => k_prev / (2.0f64.sqrt()),
                 None => {
                      // Should only happen if compute_initial_layout didn't calculate k_L
                      // Recalculate k_L based on the coarsest graph size as a fallback
                       let num_coarse = coarsest_graph.node_count() as f64;
                       let layout_width = num_coarse.sqrt() * 10.0;
                       layout_width / num_coarse.sqrt()
                 }
            };


            let mapping = &mappings[level]; // Mapping from fine (level) to coarse (level+1)

            let interpolated_layout = self.interpolate_layout(
                fine_graph,
                &current_layout, // Layout of the coarser graph (level+1)
                mapping,
                k_l, // Pass k_l for perturbation scaling
            );

            // Refine the layout at the current (finer) level
            current_layout = self.refine_layout(
                fine_graph,
                interpolated_layout,
                &self.optimizer_config, // Use standard refinement config
                current_k, // Pass the k from the previous (coarser) level
            );
             // Update k for the next iteration (the k calculated for *this* level)
             current_k = Some(k_l);
        }

        // --- Final Output ---
        // `current_layout` now holds the layout for G_0, which corresponds
        // directly to the original graph nodes because we tracked original_indices
        // and G_0 structure matches the input graph.
        assert_eq!(
            current_layout.len(),
            graph.node_count(),
            "Output layout size mismatch"
        );

        let total_duration = start_time.elapsed();
        println!("MultiLevelLayout finished in {:?}.", total_duration);

        current_layout
    }
}


// --- Example Usage (Illustrative) ---
#[cfg(test)]
mod tests {
    use super::*; // Import items from outer module

    #[test]
    fn test_simple_layout() {
        // 1. Create a simple graph
        let mut graph = SimpleGraph::<(), ()>::new(); // No user data needed
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());
        graph.add_edge(n2, n3, ());
        graph.add_edge(n3, n0, ()); // Square
        graph.add_edge(n0, n2, ()); // Diagonal

        // 2. Configure the layout algorithm
        let layout_config = MultiLevelLayout::<(), (), 2>::new( // 2D layout
            CoarseningStrategy::WalshawSmallestWeight, // Dummy for now
            10,    // Coarsening threshold
            0.95,  // Min coarsening ratio
            ForceModel::WalshawModifiedFR { repulsive_constant: 0.8 },
            0.7,   // Barnes-Hut theta (dummy tree ignores this)
            OptimizerConfig::AdaptiveGradientDescent {
                initial_temp: 50.0, // Adjust based on expected layout size
                cooling_factor: 0.90,
                tolerance: 0.05,
                max_iterations: 50, // Iterations per refinement level
            },
            100, // Iterations for initial coarsest layout
        );

        // 3. Compute the layout
        let layout = layout_config.layout(&graph);

        // 4. Print or verify the layout
        println!("Computed Layout:");
        for (i, point) in layout.iter().enumerate() {
            println!("Node {}: ({:.2}, {:.2})", i, point.x, point.y);
        }

        assert_eq!(layout.len(), graph.node_count());
        // Add more specific assertions if expected positions are known (unlikely for FDP)
        // Check that points are not all identical (unless graph is trivial)
        if layout.len() > 1 {
             assert!(layout.iter().any(|p| (p.x - layout[0].x).abs() > 1e-6 || (p.y - layout[0].y).abs() > 1e-6 ), "Layout seems collapsed.");
        }
    }
}