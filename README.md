# Circonomit - Industrial Decision Support System

A comprehensive AI-powered decision support system designed for industrial companies to understand and manage complex relationships within their operations. The system combines knowledge extraction, data-driven models, and mathematical simulations to support real business decisions.

## Overview

This project addresses the challenges faced by medium-sized industrial companies in understanding dependencies between supply chains, production processes, cost structures, and external factors like tariffs and energy costs. The solution provides a computational layer and knowledge layer that transforms complex business relationships into actionable insights.

## Project Structure

```
├── streamlit_app/          # Interactive web application for knowledge extraction
├── task1_simulation/       # Simulation engine and feedback modeling
├── task3_extraction/       # Natural language to structured data extraction
├── data/                   # Base datasets and sample data
└── requirements.txt        # Project dependencies
```

## Core Components

### 1. Simulation and Feedback Modeling (`task1_simulation/`)

**Purpose**: Advanced simulation engine that enables complex business scenario modeling with temporal effects and feedback loops.

**Key Features**:
- Dynamic input override for simulation runs (e.g., future tariff scenarios)
- Time-delayed effect modeling and temporal resolution
- Feedback loop and cyclic calculation support
- Version control and comparison of simulation results
- Graph-based representation of business relationships

**Files**:
- `simulate.py` - Core simulation engine with temporal and feedback capabilities
- `create_clean_graph.py` - Graph construction and visualization utilities
- `simulation_demo.ipynb` - Interactive demonstration of simulation features
- `feedback_graph_representation.png` - Visual representation of feedback mechanisms

**Technical Approach**:
The simulation uses a block-attribute data model where blocks group related attributes, and attributes can be inputs or calculated fields. The system supports complex interdependencies and can model scenarios like "if inventory is empty, demand shifts to alternative suppliers."

### 2. Language to Structure Extraction (`task3_extraction/`)

**Purpose**: Transforms natural language business rules into structured computational models using advanced NLP techniques.

**Key Features**:
- Rule-based and AI-driven knowledge extraction
- Structured schema validation using Pydantic
- Support for complex business logic patterns
- Integration with simulation models

**Files**:
- `schema.py` - Pydantic models for structured business rule representation
- `extraction_demo.ipynb` - Interactive extraction workflow demonstration
- `sample_rules.json` - Example business rules in structured format
- `prompt_template.txt` - Template for AI-powered extraction
- `extraction_readme.md` - Detailed extraction methodology

**Example Transformations**:
```
Input: "If energy costs rise above €200/MWh, we will postpone production by one week."
Output: Structured rule with condition blocks, thresholds, actions, and temporal delays
```

### 3. Interactive Web Application (`streamlit_app/`)

**Purpose**: User-friendly interface for real-time knowledge extraction and business rule modeling.

**Key Features**:
- Natural language input processing
- Real-time structured output generation
- Graph relationship visualization
- Integration with simulation engine
- Professional, emoji-free interface

**Files**:
- `app.py` - Main Streamlit application
- `requirements.txt` - Application-specific dependencies
- `README.md` - Application documentation

**Capabilities**:
- Parse complex business rules from natural language
- Generate structured JSON representations
- Visualize block-attribute relationships
- Provide confidence metrics for extractions

## Technical Stack

### Backend
- **Python 3.9+** - Core programming language
- **Pydantic** - Data validation and schema management
- **NetworkX** - Graph-based modeling and analysis
- **Pandas** - Data manipulation and analysis

### Frontend
- **Streamlit** - Interactive web application framework
- **Matplotlib** - Data visualization and plotting

### AI/ML
- **Natural Language Processing** - Rule extraction and pattern recognition
- **Graph Theory** - Relationship modeling and feedback analysis
- **Time Series Modeling** - Temporal effect simulation

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Interactive Application
```bash
cd streamlit_app
python -m streamlit run app.py
```
Access the application at `http://localhost:8501`

### Running Simulations
```python
from task1_simulation.simulate import run_simulation
from task1_simulation.create_clean_graph import create_business_graph

# Create and run a simulation
results = run_simulation(scenario_parameters)
```

### Extract Knowledge from Text
```python
from task3_extraction.schema import BusinessRule

# Extract structured rules from natural language
rule = extract_business_rule("If energy costs exceed €200/MWh, delay production by 1 week")
```

## Use Cases

### Supply Chain Optimization
Model complex supplier relationships and identify bottlenecks under various scenarios.

### Energy Cost Management
Simulate the impact of fluctuating energy prices on production schedules and costs.

### Risk Assessment
Analyze cascading effects of supply disruptions on overall business operations.

### Strategic Planning
Test "what-if" scenarios for strategic business decisions with quantified outcomes.

## Architecture Highlights

### Scalable Design
- Modular architecture supporting independent component development
- Caching mechanisms for efficient repeated simulations
- Parallelization strategies for complex calculations

### Knowledge Integration
- Seamless integration between natural language inputs and computational models
- Ontology-driven approach to business concept mapping
- Hybrid rule-based and AI-driven extraction methods

### User Experience
- Intuitive interface requiring no programming knowledge
- Visual feedback for model relationships and dependencies
- Explainable AI for transparent decision support

## Contributing

The system is designed for extensibility. Key areas for enhancement include:

- Additional business domain ontologies
- Advanced temporal modeling capabilities
- Enhanced visualization components
- Integration with external data sources

## Technical Notes

The project demonstrates advanced concepts in:
- **Temporal Modeling**: Handling time-delayed effects and cyclic dependencies
- **Knowledge Extraction**: Converting unstructured expert knowledge into computational models
- **Graph Theory**: Representing complex business relationships as computational graphs
- **Simulation Engineering**: Building robust, scalable simulation frameworks

---

*This project showcases the intersection of AI, business intelligence, and industrial engineering to create practical decision support tools for complex business environments.* 