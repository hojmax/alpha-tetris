//! MCTS Tree Export
//!
//! Functions for exporting the MCTS tree structure for visualization.

use super::nodes::{ChanceNode, DecisionNode};
use super::results::TreeNodeExport;

/// Recursively export a decision node and its subtree
pub(super) fn export_decision_node(
    node: &DecisionNode,
    parent_id: Option<usize>,
    edge_from_parent: Option<usize>,
    nodes: &mut Vec<TreeNodeExport>,
) -> usize {
    let id = nodes.len();
    // Create the node (children will be filled in later)
    let export = TreeNodeExport {
        id,
        node_type: "decision".to_string(),
        visit_count: node.visit_count,
        value_sum: node.value_sum,
        mean_value: node.mean_value(),
        value_history: node.value_history.clone().unwrap_or_default(),
        nn_value: node.nn_value,
        unvisited_child_value_estimate: node.initial_total_value_estimate,
        is_terminal: node.is_terminal,
        move_number: node.state.placement_count,
        attack: 0,
        state: node.state.clone(),
        parent_id,
        edge_from_parent,
        children: Vec::new(),
        valid_actions: node.valid_actions.clone(),
        action_priors: node.action_priors.clone(),
    };

    nodes.push(export);

    // Export children (sorted by action index for deterministic order)
    let mut children: Vec<(usize, &ChanceNode)> = node.children().collect();
    children.sort_by_key(|&(action_idx, _)| action_idx);

    let mut child_ids = Vec::new();
    for (action_idx, chance_node) in children {
        let child_id = export_chance_node(chance_node, Some(id), Some(action_idx), nodes);
        child_ids.push(child_id);
    }

    // Update our children list
    nodes[id].children = child_ids;

    id
}

/// Recursively export a chance node and its subtree
pub(super) fn export_chance_node(
    node: &ChanceNode,
    parent_id: Option<usize>,
    edge_from_parent: Option<usize>,
    nodes: &mut Vec<TreeNodeExport>,
) -> usize {
    let id = nodes.len();

    let export = TreeNodeExport {
        id,
        node_type: "chance".to_string(),
        visit_count: node.visit_count,
        value_sum: node.value_sum,
        mean_value: node.mean_value(),
        value_history: node.value_history.clone().unwrap_or_default(),
        nn_value: 0.0,
        unvisited_child_value_estimate: 0.0,
        is_terminal: false,
        move_number: node.state.placement_count,
        attack: node.attack,
        state: node.state.clone(),
        parent_id,
        edge_from_parent,
        children: Vec::new(),
        valid_actions: Vec::new(),
        action_priors: Vec::new(),
    };

    nodes.push(export);

    // Export children (ascending piece type / chance outcome key)
    let mut child_ids = Vec::new();
    for (piece_type, decision_node) in node.children() {
        let child_id = export_decision_node(decision_node, Some(id), Some(piece_type), nodes);
        child_ids.push(child_id);
    }

    nodes[id].children = child_ids;

    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::env::TetrisEnv;
    use crate::search::nodes::ChanceOutcomes;

    #[test]
    fn test_export_decision_node_sorts_children_and_sets_edges() {
        let env = TetrisEnv::with_seed(10, 20, 7);
        let mut root = DecisionNode::new(env.clone());
        root.visit_count = 4;
        root.value_sum = 10.0;

        let chance_high = ChanceNode::new(env.clone(), 3, 0, ChanceOutcomes::from_keys(&[1, 2]));
        let mut chance_low = ChanceNode::new(env.clone(), 1, 0, ChanceOutcomes::from_keys(&[4, 5]));

        // Insert out of order to verify deterministic sorting in export.
        let mut env_piece_5 = env.clone();
        env_piece_5.placement_count = 1;
        let mut decision_piece_5 = DecisionNode::new(env_piece_5);
        decision_piece_5.visit_count = 2;
        decision_piece_5.value_sum = 5.0;
        let mut env_piece_2 = env.clone();
        env_piece_2.placement_count = 1;
        let mut decision_piece_2 = DecisionNode::new(env_piece_2);
        decision_piece_2.visit_count = 1;
        decision_piece_2.value_sum = 1.0;
        chance_low.set_child(5, decision_piece_5);
        chance_low.set_child(2, decision_piece_2);

        let low_action = root.valid_actions[1];
        let high_action = root.valid_actions[10];
        root.set_child_for_action(high_action, chance_high);
        root.set_child_for_action(low_action, chance_low);

        let mut nodes = Vec::new();
        let root_id = export_decision_node(&root, None, None, &mut nodes);

        assert_eq!(root_id, 0);
        assert_eq!(nodes[0].node_type, "decision");
        assert_eq!(nodes[0].mean_value, 2.5);

        // Root children should be sorted by action index.
        let root_child_edges: Vec<usize> = nodes[0]
            .children
            .iter()
            .map(|&id| nodes[id].edge_from_parent.unwrap())
            .collect();
        assert!(low_action < high_action);
        assert_eq!(root_child_edges, vec![low_action, high_action]);

        // The chance node on the lower action should sort its piece children: 2 then 5.
        let chance_node_id = nodes[0].children[0];
        let chance_child_edges: Vec<usize> = nodes[chance_node_id]
            .children
            .iter()
            .map(|&id| nodes[id].edge_from_parent.unwrap())
            .collect();
        assert_eq!(chance_child_edges, vec![2, 5]);
    }
}
