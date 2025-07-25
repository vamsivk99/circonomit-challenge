#!/usr/bin/env python3
"""
Clean graph-based representation of STK Production data model with feedback loops.

This addresses the core question: "How would you represent feedback in a graph-based data model?"
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

def create_clean_graph():
    """Create a clean, professional graph representation."""
    
    # Load the data model
    df = pd.read_csv('../data/base_data.csv')
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with metadata
    for _, row in df.iterrows():
        node_id = f"{row['block']}.{row['attribute']}"
        G.add_node(node_id, 
                  block=row['block'],
                  attribute=row['attribute'],
                  type=row['type'],
                  formula=row['formula'])
    
    # Add edges based on dependencies
    for _, row in df[df['type'] == 'calc'].iterrows():
        target = f"{row['block']}.{row['attribute']}"
        formula = str(row['formula'])
        
        # Parse dependencies from formula
        dependencies = []
        
        # Cross-block references (Block.attribute)
        import re
        cross_refs = re.findall(r'\b([A-Z][a-zA-Z]*\.[a-zA-Z_][a-zA-Z0-9_]*)\b', formula)
        dependencies.extend(cross_refs)
        
        # Same-block references
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
        for word in words:
            if word not in ['if', 'else', 'max', 'min', 'and', 'or', 'not']:
                # Check if it's an attribute in any block
                for _, r in df.iterrows():
                    if r['attribute'] == word:
                        same_ref = f"{r['block']}.{word}"
                        if same_ref not in dependencies:
                            dependencies.append(same_ref)
                        break
        
        # Special handling for feedback (backlog_prev)
        if 'backlog_prev' in formula:
            # This creates a feedback edge from backlog to current node
            for _, r in df.iterrows():
                if r['attribute'] == 'backlog':
                    backlog_node = f"{r['block']}.backlog"
                    dependencies.append(backlog_node)
                    break
        
        # Add edges
        for dep in dependencies:
            if dep in G.nodes():
                G.add_edge(dep, target)
    
    return G, df

def visualize_feedback_graph(G, df):
    """Create a clean visualization focused on feedback representation."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define layout with better spacing
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # Group nodes by block for visual grouping
    blocks = {'Energy': [], 'Production': [], 'Demand': [], 'Inventory': []}
    for node in G.nodes():
        if '.' in node:
            block_name = node.split('.')[0]
            if block_name in blocks:
                blocks[block_name].append(node)
    
    # Define block colors and positions for grouping rectangles
    block_colors = {
        'Energy': '#FFE5E5',      # Light red
        'Production': '#E5F7F7',  # Light teal  
        'Demand': '#E5F0FF',      # Light blue
        'Inventory': '#F0FFE5'    # Light green
    }
    
    # Draw block grouping rectangles
    for block_name, nodes in blocks.items():
        if nodes:
            # Calculate bounding box for this block's nodes
            x_coords = [pos[node][0] for node in nodes]
            y_coords = [pos[node][1] for node in nodes]
            
            min_x, max_x = min(x_coords) - 0.15, max(x_coords) + 0.15
            min_y, max_y = min(y_coords) - 0.15, max(y_coords) + 0.15
            
            # Create rounded rectangle for block grouping
            from matplotlib.patches import FancyBboxPatch
            rect = FancyBboxPatch(
                (min_x, min_y), max_x - min_x, max_y - min_y,
                boxstyle="round,pad=0.02",
                facecolor=block_colors[block_name],
                edgecolor='#CCCCCC',
                linewidth=1.5,
                alpha=0.3
            )
            ax.add_patch(rect)
            
            # Add block label with larger font
            center_x = (min_x + max_x) / 2
            ax.text(center_x, max_y + 0.05, f"{block_name} Block", 
                   ha='center', va='bottom', fontsize=12, fontweight='bold',
                   color='#2C3E50', alpha=0.8)
    
    # Classify edges into regular and feedback
    feedback_edges = []
    regular_edges = []
    
    for edge in G.edges():
        source, target = edge
        if ('backlog' in source and 'demand' in target) or \
           ('stock' in source and 'demand' in target) or \
           ('demand' in source and 'backlog' in target):
            feedback_edges.append(edge)
        else:
            regular_edges.append(edge)
    
    # Draw regular edges with better routing
    for edge in regular_edges:
        source, target = edge
        # Use different curve radii to avoid overlaps
        if 'price_future' in source and 'demand_shift' in target:
            connectionstyle = "arc3,rad=0.3"  # More curved to avoid feedback overlap
        elif 'price_future' in target:
            connectionstyle = "arc3,rad=0.2"  # More curved for busy nodes
        elif 'CO2_factor' in source and 'unit_emissions' in target:
            connectionstyle = "arc3,rad=0.25"  # Ensure CO2 edge is visible
        else:
            connectionstyle = "arc3,rad=0.1"
            
        nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                              edge_color='#2C3E50', arrows=True, 
                              arrowsize=20, width=2, alpha=0.7,
                              connectionstyle=connectionstyle)
        


    
    # Draw feedback edges with special styling
    for edge in feedback_edges:
        source, target = edge
        # Special handling for self-loops
        if source == target:
            connectionstyle = "arc3,rad=0.9"  # Large curve for self-loop
        else:
            connectionstyle = "arc3,rad=0.4"  # Prominent curve for feedback
            
        nx.draw_networkx_edges(G, pos, edgelist=[edge],
                              edge_color='#E74C3C', arrows=True,
                              arrowsize=25, width=3, alpha=0.9,
                              style='dashed', connectionstyle=connectionstyle)
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        if node_type == 'input':
            node_colors.append('#27AE60')  # Green for inputs
        else:
            node_colors.append('#3498DB')  # Blue for calculations
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1500, alpha=0.9,
                          edgecolors='white', linewidths=2)
    
    # Add labels
    labels = {}
    for node in G.nodes():
        if '.' in node:
            block, attr = node.split('.', 1)
            labels[node] = attr.replace('_', '\n')  # Remove block prefix, keep attribute only
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#27AE60', label='Input Attributes'),
        mpatches.Patch(color='#3498DB', label='Calculated Attributes'),
        plt.Line2D([0], [0], color='#2C3E50', linewidth=2, label='Dependencies'),
        plt.Line2D([0], [0], color='#E74C3C', linewidth=3, 
                  linestyle='--', label='Feedback Loops')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    ax.set_title('STK Production: Wirkungsnetz (Impact Network)\nGraph-Based Data Model with Feedback Representation', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add technical explanation as caption
    caption_text = ('Edges represent dependencies stored as <source_attr> → <target_attr> triples.\n'
                   'Blocks become node labels in property graph; cycles are first-class citizens.\n'
                   'Dashed red edges represent feedback cycles in the system.')
    
    ax.text(0.5, 0.02, caption_text, 
           ha='center', va='bottom', transform=ax.transAxes, 
           fontsize=9, style='italic', color='#2C3E50')
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save with proper Circonomit naming
    plt.savefig('../diagrams/wirkungsnetz_graph_model.png', dpi=300, bbox_inches='tight')
    plt.savefig('../diagrams/wirkungsnetz_graph_model.svg', format='svg', bbox_inches='tight')
    
    # Save PDF-ready version (smaller, optimized for documents)
    plt.savefig('../diagrams/wirkungsnetz_graph_model_pdf.png', dpi=200, 
                bbox_inches='tight', facecolor='white')
    
    # Also save local copy
    plt.savefig('feedback_graph_representation.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feedback_structure(G):
    """Analyze and document the feedback structure."""
    
    print("Feedback Analysis in Graph-Based Data Model")
    print("=" * 50)
    
    # Check for cycles
    has_cycles = not nx.is_directed_acyclic_graph(G)
    print(f"Contains feedback loops: {has_cycles}")
    
    if has_cycles:
        print("\nIdentified feedback loops:")
        try:
            cycles = list(nx.simple_cycles(G))
            for i, cycle in enumerate(cycles, 1):
                print(f"  Cycle {i}: {' -> '.join(cycle)} -> {cycle[0]}")
        except:
            print("  Complex feedback structure detected")
    
    print(f"\nGraph statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    return has_cycles

if __name__ == "__main__":
    G, df = create_clean_graph()
    analyze_feedback_structure(G)
    
    # Print all edges to verify CO2 dependency
    print(f"\nAll {G.number_of_edges()} edges in the graph:")
    for i, edge in enumerate(G.edges(), 1):
        source, target = edge
        if 'CO2_factor' in source and 'unit_emissions' in target:
            print(f"{i:2d}. {edge[0]} → {edge[1]} ✓ (CO2 dependency)")
        elif any(keyword in edge[0] and keyword in edge[1] for keyword in ['backlog', 'demand', 'stock']):
            print(f"{i:2d}. {edge[0]} → {edge[1]} ⭕ (feedback)")
        else:
            print(f"{i:2d}. {edge[0]} → {edge[1]}")
    
    visualize_feedback_graph(G, df)
    print("\nFinal polished visualizations saved:")
    print("- diagrams/wirkungsnetz_graph_model.svg (vector)")
    print("- diagrams/wirkungsnetz_graph_model.png (high-res)")
    print("- diagrams/wirkungsnetz_graph_model_pdf.png (PDF-ready)")
    print("- feedback_graph_representation.png (local copy)") 