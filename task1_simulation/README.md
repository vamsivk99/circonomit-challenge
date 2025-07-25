# Task 1: Simulation and Feedback in Data Models

## Problem Statement

Extend the existing block-attribute data model to enable simulation runs with:
- Input overrides for scenario testing
- Time-delayed effects 
- Feedback loops and cycles
- Results storage and comparison
- Graph-based representation of feedback

## Solution Approach

### Data Model Structure

The system uses a CSV-based data model with the following structure:
- **Blocks**: Logical groupings (Energy, Production, Demand, Inventory)
- **Attributes**: Either inputs or calculated fields within blocks
- **Dependencies**: Calculated fields reference other attributes via formulas

### Feedback Representation in Graph Model

**Core Question**: How to represent feedback in a graph-based data model?

**Answer**: Using a directed graph where:
1. **Nodes** represent attributes (inputs and calculations)
2. **Edges** represent dependencies between attributes  
3. **Feedback loops** are represented as cycles in the graph
4. **Time delays** create temporal feedback through lagged references

### Implementation Details

#### Feedback Mechanisms
1. **Cross-temporal feedback**: `backlog_prev` creates dependency on previous time step
2. **Cross-block feedback**: Inventory levels affect demand decisions
3. **Conditional feedback**: Demand shifts based on price thresholds

#### Graph Construction
```python
# Nodes: Each attribute becomes a node
G.add_node("Energy.price_future", type="calc", formula="...")

# Edges: Dependencies from formula parsing
G.add_edge("Energy.price_now", "Energy.price_future")

# Feedback: Special handling for temporal references
if 'backlog_prev' in formula:
    G.add_edge("Inventory.backlog", current_node)  # Creates cycle
```

#### Cycle Detection
- NetworkX `is_directed_acyclic_graph()` detects feedback presence
- `simple_cycles()` identifies specific feedback loops
- Visual distinction: dashed red edges for feedback

### Simulation Engine

#### Input Overrides
```python
simulate(model, overrides={'Energy.tariff_future': 0.3})
```

#### Time-Delayed Effects
- `lag` parameter in data model
- Previous step value storage
- Formula evaluation with temporal references

#### Results Storage
- Per-timestep DataFrames
- Multi-scenario comparison
- Export capabilities

## Files Structure

- `../data/base_data.csv` - Data model definition
- `simulate.py` - Main simulation engine
- `create_clean_graph.py` - Clean graph visualization
- `simulation_demo.ipynb` - Interactive demonstration

## Usage

```bash
cd task1_simulation
python create_clean_graph.py
```

This generates:
- `feedback_graph_representation.png` - Clean graph visualization
- Console output with feedback analysis

## Key Results

The implementation successfully creates a graph-based data model where:
- Feedback loops are explicitly represented as graph cycles
- Time delays enable realistic business dynamics
- Scenario testing supports decision-making
- Visual representation clearly shows system structure

The approach demonstrates that feedback in graph-based data models can be effectively represented through directed cycles, with temporal aspects handled via lagged node references. 