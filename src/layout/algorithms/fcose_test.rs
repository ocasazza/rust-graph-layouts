#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Graph, Node, Edge};
    use std::collections::HashMap;

    fn create_test_graph() -> Graph {
        let mut nodes = HashMap::new();
        let mut edges = HashMap::new();

        // Add two nodes with fixed positions
        let mut node1 = Node::default();
        node1.position = Some((0.0, 0.0));
        nodes.insert("1".to_string(), node1);

        let mut node2 = Node::default();
        node2.position = Some((100.0, 0.0));
        nodes.insert("2".to_string(), node2);

        // Add an edge between them
        let edge = Edge {
            source: "1".to_string(),
            target: "2".to_string(),
            weight: 1.0,
        };
        edges.insert("1-2".to_string(), edge);

        Graph { nodes, edges }
    }

    #[test]
    fn test_fcose_options_default() {
        let options = FcoseOptions::default();
        assert_eq!(options.quality, "default");
        assert_eq!(options.node_repulsion, 4500.0);
        assert_eq!(options.ideal_edge_length, 50.0);
        assert_eq!(options.node_overlap, 10.0);
    }

    #[test]
    fn test_repulsion_force_calculation() {
        let graph = create_test_graph();
        let options = FcoseOptions::default();
        let engine = FcoseLayoutEngine::new(options);
        
        let forces = engine.calculate_repulsion(&graph);
        
        // For two nodes 100 units apart, we can calculate the expected repulsion
        // Force = node_repulsion / distance^2
        let expected_force = 4500.0 / (100.0 * 100.0);
        let force_magnitude = (forces[0].0 * forces[0].0 + forces[0].1 * forces[0].1).sqrt();
        
        // Allow for some floating point imprecision
        assert!((force_magnitude - expected_force).abs() < 0.001);
    }

    #[test]
    fn test_attraction_force_calculation() {
        let graph = create_test_graph();
        let options = FcoseOptions::default();
        let engine = FcoseLayoutEngine::new(options);
        
        let forces = engine.calculate_attraction(&graph);
        
        // For two nodes 100 units apart with ideal length 50:
        // Force = (distance - ideal_length) / 3
        let expected_force = (100.0 - 50.0) / 3.0;
        let force_magnitude = (forces[0].0 * forces[0].0 + forces[0].1 * forces[0].1).sqrt();
        
        // Allow for some floating point imprecision
        assert!((force_magnitude - expected_force).abs() < 0.001);
    }

    #[test]
    fn test_overlap_removal() {
        let mut graph = Graph {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        };

        // Create two overlapping nodes
        let mut node1 = Node::default();
        node1.position = Some((0.0, 0.0));
        graph.nodes.insert("1".to_string(), node1);

        let mut node2 = Node::default();
        node2.position = Some((5.0, 0.0));  // Very close to node1
        graph.nodes.insert("2".to_string(), node2);

        let options = FcoseOptions::default();
        let engine = FcoseLayoutEngine::new(options);
        
        // Remove overlaps
        engine.remove_overlaps(&mut graph).unwrap();

        // Get final positions
        let pos1 = graph.nodes.get("1").unwrap().position.unwrap();
        let pos2 = graph.nodes.get("2").unwrap().position.unwrap();
        
        // Calculate final distance
        let dx = pos2.0 - pos1.0;
        let dy = pos2.1 - pos1.1;
        let final_distance = (dx * dx + dy * dy).sqrt();
        
        // Nodes should be at least min_distance apart (20.0 * (1.0 - 10.0/100.0))
        let min_distance = 20.0 * (1.0 - 10.0/100.0);
        assert!(final_distance >= min_distance);
    }
}
