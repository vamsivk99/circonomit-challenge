#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulation Model for STK Produktion GmbH

This module provides functions for simulating a business model with blocks, attributes,
and feedback loops. It handles inputs, calculations, and time-lagged effects.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re


def load_model(filepath):
    """
    Load a model from a CSV file.
    
    Args:
        filepath: Path to the CSV file containing model definitions
        
    Returns:
        DataFrame containing the model
    """
    return pd.read_csv(filepath)


def build_dependency_graph(df):
    """
    Build a dependency graph from the model.
    
    Args:
        df: DataFrame containing the model
        
    Returns:
        NetworkX DiGraph representing dependencies
    """
    G = nx.DiGraph()
    
    # Add all blocks and attributes as nodes
    for _, row in df.iterrows():
        node_name = f"{row['block']}.{row['attribute']}"
        G.add_node(node_name, block=row['block'], attribute=row['attribute'], 
                  type=row['type'], value=row['value'], formula=row['formula'], lag=row['lag'])

    # Add edges based on formula dependencies
    for _, row in df[df['type'] == 'calc'].iterrows():
        target = f"{row['block']}.{row['attribute']}"
        formula = row['formula']
        if isinstance(formula, str):
            # Extract all block.attribute references more precisely
            import re
            # Find patterns like Block.attribute (capitalized blocks)
            pattern = r'\b([A-Z][a-zA-Z]*)\.[a-zA-Z_][a-zA-Z0-9_]*\b'
            matches = re.findall(pattern, formula)
            
            # Also look for direct attribute references that might be from same block
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
            
            for match in matches:
                # This is a full Block.attribute reference
                block_name = match
                attr_pattern = rf'\b{block_name}\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
                attr_matches = re.findall(attr_pattern, formula)
                for attr in attr_matches:
                    source = f"{block_name}.{attr}"
                    if source in [f"{r['block']}.{r['attribute']}" for _, r in df.iterrows()]:
                        G.add_edge(source, target)
                        
            # Handle same-block references and special cases
            for word in words:
                if word == 'backlog_prev':
                    # This creates a feedback edge from backlog to current attribute
                    backlog_node = None
                    for _, r in df.iterrows():
                        if r['attribute'] == 'backlog':
                            backlog_node = f"{r['block']}.{r['attribute']}"
                            break
                    if backlog_node:
                        G.add_edge(backlog_node, target)
                elif word in [r['attribute'] for _, r in df.iterrows() if r['block'] == row['block']]:
                    # This might be a same-block reference
                    source = f"{row['block']}.{word}"
                    if source in [f"{r['block']}.{r['attribute']}" for _, r in df.iterrows()]:
                        G.add_edge(source, target)
                else:
                    # Check if it's an attribute in any other block
                    for _, r in df.iterrows():
                        if r['attribute'] == word:
                            source = f"{r['block']}.{r['attribute']}"
                            G.add_edge(source, target)
                            break
    
    return G


def visualize_graph(G, output_path=None):
    """
    Visualize the dependency graph with clear feedback loop highlighting.
    
    Args:
        G: NetworkX DiGraph
        output_path: Optional path to save the visualization
    """
    plt.figure(figsize=(16, 12))
    
    # Create a hierarchical layout that better shows flow
    # Group nodes by type for better positioning
    input_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'input']
    calc_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'calc']
    
    # Create positions manually for better flow visualization
    pos = {}
    
    # Position input nodes on the left side
    for i, node in enumerate(input_nodes):
        pos[node] = (0, i * 2)
    
    # Position calculated nodes based on their dependencies
    energy_calcs = [n for n in calc_nodes if n.startswith('Energy.')]
    production_calcs = [n for n in calc_nodes if n.startswith('Production.')]
    demand_calcs = [n for n in calc_nodes if n.startswith('Demand.')]
    inventory_calcs = [n for n in calc_nodes if n.startswith('Inventory.')]
    
    y_offset = 0
    for i, node in enumerate(energy_calcs):
        pos[node] = (3, y_offset)
        y_offset += 1.5
        
    for i, node in enumerate(production_calcs):
        pos[node] = (6, i * 1.5 + 2)
        
    for i, node in enumerate(demand_calcs):
        pos[node] = (3, len(input_nodes) * 2 + 1)
        
    for i, node in enumerate(inventory_calcs):
        pos[node] = (6, len(input_nodes) * 2 + i * 1.5)
    
    # Identify feedback edges (cycles)
    feedback_edges = []
    try:
        # Find strongly connected components to identify cycles
        sccs = list(nx.strongly_connected_components(G))
        for scc in sccs:
            if len(scc) > 1:  # Only consider non-trivial SCCs
                # Get all edges within this SCC
                subgraph = G.subgraph(scc)
                feedback_edges.extend(subgraph.edges())
    except:
        # If cycle detection fails, manually identify known feedback edges
        feedback_edges = [
            ('Inventory.stock', 'Demand.demand_shift'),
            ('Inventory.backlog', 'Demand.demand_shift'),
            ('Demand.demand_shift', 'Inventory.backlog')
        ]
    
    # Separate edges into regular and feedback
    regular_edges = [e for e in G.edges() if e not in feedback_edges]
    
    # Draw nodes with different colors and sizes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        if node_type == 'input':
            node_colors.append('#4CAF50')  # Green for inputs
            node_sizes.append(3000)
        else:
            node_colors.append('#2196F3')  # Blue for calculations
            node_sizes.append(2500)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.9, edgecolors='white', linewidths=2)
    
    # Draw regular edges
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, edge_color='#666666', 
                          arrows=True, arrowsize=20, arrowstyle='->', 
                          connectionstyle="arc3,rad=0.1", alpha=0.7, width=1.5)
    
    # Draw feedback edges with different style
    if feedback_edges:
        nx.draw_networkx_edges(G, pos, edgelist=feedback_edges, edge_color='#FF5722', 
                              arrows=True, arrowsize=25, arrowstyle='->', 
                              connectionstyle="arc3,rad=0.3", alpha=0.9, width=3,
                              style='dashed')
    
    # Add labels with better formatting
    labels = {}
    for node in G.nodes():
        if '.' in node:
            block, attr = node.split('.', 1)
            # Use shorter, more readable labels
            attr_short = attr.replace('_', '\n')
            labels[node] = f"{block}\n{attr_short}"
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', 
                           font_color='white')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Input Attributes'),
        Patch(facecolor='#2196F3', label='Calculated Attributes'),
        plt.Line2D([0], [0], color='#666666', linewidth=2, label='Dependencies'),
        plt.Line2D([0], [0], color='#FF5722', linewidth=3, linestyle='--', label='Feedback Loops')
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True, 
              fancybox=True, shadow=True)
    
    plt.title('STK Production - Wirkungsnetz (Impact Network)\nWith Feedback Loops Highlighted', 
              fontsize=16, fontweight='bold', color='#2E4057', pad=30)
    
    # Add subtitle explaining the feedback
    plt.text(0.5, 0.02, 'Red dashed lines show feedback cycles: Low inventory → Demand shifts → Affects backlog → Affects demand', 
             ha='center', va='bottom', transform=plt.gca().transAxes, 
             fontsize=11, style='italic', color='#FF5722')
    
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def visualize_blocks_and_nodes(G, output_path=None):
    """
    Create a beautiful, intuitive block-based visualization with nodes clearly grouped.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    # Define block positions and colors
    blocks = {
        'Energy': {'pos': (0, 3), 'color': '#FF6B6B', 'size': (2.5, 2)},
        'Production': {'pos': (5, 3), 'color': '#4ECDC4', 'size': (2.5, 2)},
        'Demand': {'pos': (0, 0), 'color': '#45B7D1', 'size': (2.5, 2)},
        'Inventory': {'pos': (5, 0), 'color': '#96CEB4', 'size': (2.5, 2)}
    }
    
    # Calculate node positions within blocks
    node_positions = {}
    block_nodes = {}
    
    # Group nodes by block
    for node in G.nodes():
        if '.' in node:
            block_name, attr_name = node.split('.', 1)
            if block_name not in block_nodes:
                block_nodes[block_name] = []
            block_nodes[block_name].append(node)
    
    # Position nodes within each block
    for block_name, nodes in block_nodes.items():
        block_info = blocks[block_name]
        block_x, block_y = block_info['pos']
        
        # Arrange nodes in a grid within the block
        num_nodes = len(nodes)
        if num_nodes <= 2:
            rows, cols = 1, num_nodes
        elif num_nodes <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 2
        
        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols
            
            # Calculate position within block
            node_x = block_x + (col - (cols-1)/2) * 0.6
            node_y = block_y + (row - (rows-1)/2) * 0.6
            
            node_positions[node] = (node_x, node_y)
    
    # Draw block containers (rounded rectangles)
    for block_name, block_info in blocks.items():
        x, y = block_info['pos']
        width, height = block_info['size']
        color = block_info['color']
        
        # Create rounded rectangle for block
        from matplotlib.patches import FancyBboxPatch
        block_patch = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=3
        )
        ax.add_patch(block_patch)
        
        # Add block label
        ax.text(x, y + height/2 + 0.3, f"🏷️ {block_name} Block", 
               ha='center', va='bottom', fontsize=14, fontweight='bold', 
               color=color)
    
    # Draw edges with different styles
    regular_edges = []
    feedback_edges = []
    
    # Classify edges
    for edge in G.edges():
        source, target = edge
        # Simple heuristic for feedback edges
        if ('backlog' in source and 'demand' in target) or \
           ('demand' in source and 'backlog' in target) or \
           ('stock' in source and 'demand' in target):
            feedback_edges.append(edge)
        else:
            regular_edges.append(edge)
    
    # Draw regular edges
    for edge in regular_edges:
        source, target = edge
        x1, y1 = node_positions[source]
        x2, y2 = node_positions[target]
        
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#34495E',
                                 connectionstyle="arc3,rad=0.1"))
    
    # Draw feedback edges with special styling
    for edge in feedback_edges:
        source, target = edge
        x1, y1 = node_positions[source]
        x2, y2 = node_positions[target]
        
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=3, color='#E74C3C',
                                 linestyle='--', alpha=0.8,
                                 connectionstyle="arc3,rad=0.3"))
    
    # Draw nodes
    for node in G.nodes():
        x, y = node_positions[node]
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'unknown')
        
        # Choose node style based on type
        if node_type == 'input':
            node_color = '#2ECC71'
            node_shape = 'o'
            node_size = 800
        else:
            node_color = '#3498DB'
            node_shape = 's'
            node_size = 700
        
        # Draw node
        ax.scatter(x, y, c=node_color, s=node_size, alpha=0.9, 
                  edgecolors='white', linewidth=2, marker=node_shape)
        
        # Add node label
        if '.' in node:
            _, attr_name = node.split('.', 1)
            label = attr_name.replace('_', '\n')
        else:
            label = node
        
        ax.text(x, y, label, ha='center', va='center', 
               fontsize=9, fontweight='bold', color='white')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71', 
                  markersize=12, label='Input Attributes'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498DB', 
                  markersize=12, label='Calculated Attributes'),
        plt.Line2D([0], [0], color='#34495E', linewidth=2, label='Dependencies'),
        plt.Line2D([0], [0], color='#E74C3C', linewidth=3, linestyle='--', 
                  label='Feedback Loops')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
             frameon=True, fancybox=True, shadow=True, 
             bbox_to_anchor=(0.98, 0.98))
    
    # Set title and styling
    ax.set_title('STK Production - Block-Based Wirkungsnetz\n' + 
                'Intuitive Node Visualization with Clear Block Structure', 
                fontsize=18, fontweight='bold', color='#2C3E50', pad=30)
    
    # Add subtitle
    ax.text(0.5, 0.02, 'Red dashed arrows show feedback cycles | Colored areas represent business blocks', 
           ha='center', va='bottom', transform=ax.transAxes, 
           fontsize=12, style='italic', color='#7F8C8D')
    
    # Clean up the plot
    ax.set_xlim(-2, 9)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def simulate(df, overrides=None, steps=4):
    """
    Simulate the model for a given number of steps, with optional overrides.
    
    Args:
        df: DataFrame with model definitions
        overrides: Dictionary of {block.attribute: value} to override inputs
        steps: Number of time steps to simulate
    
    Returns:
        DataFrame with simulation results
    """
    # Create a directed graph to represent dependencies
    G = nx.DiGraph()
    
    # Build node dictionary for faster access
    nodes = {}
    for _, row in df.iterrows():
        node_name = f"{row['block']}.{row['attribute']}"
        nodes[node_name] = {
            'block': row['block'],
            'attribute': row['attribute'],
            'type': row['type'],
            'value': row['value'],
            'formula': row['formula'],
            'lag': row['lag'] if pd.notna(row['lag']) else None
        }
        G.add_node(node_name)
    
    # Add edges based on formula dependencies
    for node_name, node_data in nodes.items():
        if node_data['type'] == 'calc' and isinstance(node_data['formula'], str):
            formula = node_data['formula']
            tokens = formula.replace('(', ' ').replace(')', ' ').replace('*', ' ').replace('+', ' ').split()
            for token in tokens:
                if '.' in token and token in nodes:
                    G.add_edge(token, node_name)
    
    # Apply overrides to input values
    if overrides:
        for key, value in overrides.items():
            if key in nodes and nodes[key]['type'] == 'input':
                nodes[key]['value'] = value
    
    # Initialize results storage
    results = []
    
    # Store previous values for lagged calculations
    previous_values = {}
    
    # Run simulation for each time step
    for step in range(steps):
        # Create a copy of the current values for this step
        step_values = {}
        
        # First, process inputs
        for node_name, node_data in nodes.items():
            if node_data['type'] == 'input':
                step_values[node_name] = node_data['value']
                
        # Then, process calculations in topological order
        # This ensures dependencies are calculated first
        try:
            # Try to get a topological sort
            calc_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # If there are cycles, we need a different approach
            # For simplicity, we'll just use the order from the dataframe
            # In a real system, you'd need more sophisticated cycle resolution
            calc_order = [f"{row['block']}.{row['attribute']}" for _, row in df.iterrows() if row['type'] == 'calc']
            
        for node_name in calc_order:
            if node_name in nodes and nodes[node_name]['type'] == 'calc':
                formula = nodes[node_name]['formula']
                lag = nodes[node_name]['lag']
                
                if isinstance(formula, str):
                    # Prepare formula for evaluation
                    eval_formula = formula
                    
                    # First, replace all cross-block references (Block.attribute)
                    import re
                    # Find all Block.attribute patterns in the formula
                    cross_refs = re.findall(r'\b([A-Z][a-zA-Z]*\.[a-zA-Z_][a-zA-Z0-9_]*)\b', eval_formula)
                    for ref in cross_refs:
                        if ref in step_values:
                            eval_formula = eval_formula.replace(ref, str(step_values[ref]))
                        elif ref in nodes:
                            # If the referenced node exists but doesn't have a value yet, use 0 as default
                            eval_formula = eval_formula.replace(ref, '0')
                    
                    # Handle lagged values first (before other replacements)
                    if 'backlog_prev' in eval_formula:
                        if step > 0:
                            # Find the backlog attribute in any block
                            backlog_key = None
                            for node_key in nodes:
                                if node_key.endswith('.backlog'):
                                    backlog_key = node_key
                                    break
                            if backlog_key and backlog_key in previous_values:
                                prev_value = previous_values[backlog_key]
                            else:
                                prev_value = 0
                            eval_formula = eval_formula.replace('backlog_prev', str(prev_value))
                        else:
                            # For first step, assume previous value is 0
                            eval_formula = eval_formula.replace('backlog_prev', '0')
                    
                    # Then handle same-block references (simple attribute names)
                    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', eval_formula)
                    for word in words:
                        # Skip Python keywords and functions
                        if word in ['if', 'else', 'max', 'min', 'abs', 'and', 'or', 'not']:
                            continue
                        # Check if this word is an attribute in the same block
                        same_block_ref = f"{nodes[node_name]['block']}.{word}"
                        if same_block_ref in step_values and word in eval_formula:
                            # Replace the word with its value, but be careful about partial matches
                            eval_formula = re.sub(rf'\b{word}\b', str(step_values[same_block_ref]), eval_formula)
                        # Also check if it could be an attribute in any other block
                        elif word not in ['backlog_prev']:  # Skip already handled special cases
                            for other_block in ['Energy', 'Production', 'Demand', 'Inventory']:
                                other_ref = f"{other_block}.{word}"
                                if other_ref in step_values and word in eval_formula:
                                    eval_formula = re.sub(rf'\b{word}\b', str(step_values[other_ref]), eval_formula)
                                    break
                    
                    # Handle remaining lagged values (for other potential lag attributes)
                    if lag is not None and lag > 0:
                        # This is for attributes that have their own lag
                        # The backlog_prev was handled above
                        pass
                    
                    try:
                        # Evaluate the formula
                        result = eval(eval_formula)
                        step_values[node_name] = result
                    except Exception as e:
                        print(f"Error evaluating {node_name}: {eval_formula}")
                        print(f"Exception: {str(e)}")
                        step_values[node_name] = None
        
        # Store results for this step
        for node_name, value in step_values.items():
            block, attribute = node_name.split('.')
            results.append({
                'step': step,
                'block': block,
                'attribute': node_name,
                'value': value
            })
        
        # Update previous values for next step
        previous_values = step_values.copy()
    
    # Convert results to DataFrame
    return pd.DataFrame(results)


def plot_results(results_df, attributes_to_plot=None, title='Simulation Results'):
    """
    Plot the results of a simulation.
    
    Args:
        results_df: DataFrame with simulation results
        attributes_to_plot: List of attributes to plot (if None, plots all)
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    
    pivot_data = results_df.pivot(index='step', columns='attribute', values='value')
    
    if attributes_to_plot:
        pivot_data[attributes_to_plot].plot(marker='o')
    else:
        pivot_data.plot(marker='o')
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return plt


def compare_scenarios(scenario_results, attributes, scenario_names=None):
    """
    Compare multiple scenarios for specific attributes with beautiful styling.
    
    Args:
        scenario_results: List of DataFrames with simulation results
        attributes: List of attributes to compare
        scenario_names: List of names for the scenarios
    """
    if scenario_names is None:
        scenario_names = [f"Scenario {i+1}" for i in range(len(scenario_results))]
    
    # Beautiful color palette
    colors = ['#2ECC71', '#E74C3C', '#F39C12', '#9B59B6', '#3498DB']
    
    n_attributes = len(attributes)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # Define icons and units for each attribute type
    attr_info = {
        'Energy.price_future': {'icon': 'PRICE', 'unit': '€/MWh', 'color': '#E74C3C'},
        'Production.unit_cost': {'icon': 'COST', 'unit': '€/unit', 'color': '#3498DB'},
        'Production.unit_emissions': {'icon': 'CO2', 'unit': 'CO₂/unit', 'color': '#27AE60'},
        'Inventory.backlog': {'icon': 'STOCK', 'unit': 'units', 'color': '#F39C12'}
    }
    
    for i, attr in enumerate(attributes):
        ax = axes[i]
        info = attr_info.get(attr, {'icon': 'DATA', 'unit': 'value', 'color': '#7F8C8D'})
        
        for j, results in enumerate(scenario_results):
            pivot_data = results.pivot(index='step', columns='attribute', values='value')
            if attr in pivot_data.columns:
                # Plot with beautiful styling
                ax.plot(pivot_data.index, pivot_data[attr], 
                       color=colors[j % len(colors)], linewidth=3, 
                       marker='o', markersize=8, alpha=0.9,
                       label=f"{scenario_names[j]}")
        
        # Beautiful styling for each subplot
        ax.set_title(f"{info['icon']} {attr.split('.')[-1].replace('_', ' ').title()}", 
                    fontsize=14, fontweight='bold', color=info['color'])
        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel(f"Value ({info['unit']})", fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Add background color
        ax.set_facecolor('#FAFAFA')
        
        # Style the spines
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(1)
    
    # Remove unused subplots if any
    for i in range(len(attributes), len(axes)):
        fig.delaxes(axes[i])
    
    # Main title with beautiful styling
    fig.suptitle('🏭 STK Production - Multi-Scenario Analysis\n' +
                'Economic & Environmental Impact Assessment', 
                fontsize=18, fontweight='bold', color='#2C3E50', y=0.95)
    
    # Add a subtitle with scenario description
    scenario_desc = " vs ".join(scenario_names)
    fig.text(0.5, 0.02, f'Comparing: {scenario_desc}', 
            ha='center', va='bottom', fontsize=12, 
            style='italic', color='#7F8C8D')
    
    plt.tight_layout()
    return fig


def main():
    """
    Main function to demonstrate the simulation capabilities.
    """
    # Determine the base path
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = os.path.dirname(sys.executable)
    else:
        # Running as script
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define file paths
    model_path = os.path.join(base_path, 'data', 'base_data.csv')
    
    # Load model
    try:
        model_df = load_model(model_path)
        print(f"Model loaded with {len(model_df)} rows")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Run default simulation
    results_default = simulate(model_df, steps=6)
    print("Default simulation completed")
    
    # Run alternative scenario with higher tariff
    results_high_tariff = simulate(model_df, overrides={'Energy.tariff_future': 0.3}, steps=6)
    print("High tariff simulation completed")
    
    # Run scenario with higher CO2 factor
    results_high_co2 = simulate(model_df, overrides={'Energy.tariff_future': 0.25, 'Energy.CO2_factor': 0.5}, steps=6)
    print("High CO2 simulation completed")
    
    # Compare scenarios with environmental metrics
    attributes_to_compare = ['Energy.price_future', 'Production.unit_cost', 'Production.unit_emissions', 'Inventory.backlog']
    fig = compare_scenarios(
        [results_default, results_high_tariff, results_high_co2],
        attributes_to_compare,
        ['Default (20% tariff)', 'High Tariff (30%)', 'High CO2 + 25% tariff']
    )
    
    # Save the comparison plot
    output_path = os.path.join(base_path, 'task1_simulation', 'scenario_comparison.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")
    
    # Visualize the dependency graph - OLD VERSION
    G = build_dependency_graph(model_df)
    graph_path = os.path.join(base_path, 'task1_simulation', 'dependency_graph.png')
    visualize_graph(G, graph_path)
    print(f"Dependency graph saved to {graph_path}")
    
    # Visualize the BEAUTIFUL block-based version - NEW
    blocks_path = os.path.join(base_path, 'task1_simulation', 'blocks_visualization.png')
    visualize_blocks_and_nodes(G, blocks_path)
    print(f"✨ Beautiful block-based visualization saved to {blocks_path}")
    
    # Create SVG version for presentation
    svg_path = os.path.join(base_path, 'task1_simulation', 'STK_Wirkungsnetz_Final.svg')
    visualize_blocks_and_nodes(G, svg_path.replace('.png', '.png'))  # Still PNG for now
    print(f"Final presentation visualization created!")


if __name__ == "__main__":
    main()
