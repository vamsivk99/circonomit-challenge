# Circonomit - Language to Structure Demo

Interactive Streamlit demo for Task 3: Knowledge Extraction from Natural Language

## Features

- **Live Rule Extraction**: Paste business rules, get structured JSON output
- **Sample Rules**: Pre-loaded examples from the challenge
- **Graph Visualization**: See Block-Attribute relationships
- **Integration Preview**: Shows how rules connect to Task 1 simulation

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run from project root:
```bash
streamlit run streamlit_app/app.py
```

3. Open browser to: `http://localhost:8501`

### Demo Usage

1. Select a sample rule or enter custom text
2. Click "Extract Knowledge" 
3. View structured JSON output
4. See graph relationship mapping
5. Review integration with Task 1

## Sample Rules

- **Energy Price**: "If energy costs rise above €200/MWh, we will postpone production by one week."
- **Demand Shift**: "Demand in France drops by 60% at a price above €250 per unit."
- **Customer Loss**: "If we can't deliver for two weeks, there's an 80% chance we'll lose the customer."

## Technical Stack

- **Frontend**: Streamlit with two-column layout
- **Backend**: Pydantic schema validation
- **Processing**: Mock LLM extraction (production-ready for GPT-4)
- **Visualization**: NetworkX graph relationships
- **Integration**: Task 1 Block-Attribute model

## Architecture

```
Natural Language → Schema Validation → Graph Mapping → Simulation Ready
```

The demo shows the complete pipeline from expert communication to computational model integration.

## Production Notes

For production deployment:
- Replace mock extraction with real LLM API calls
- Add authentication and user management
- Implement persistent storage for extracted rules
- Connect to live simulation engine from Task 1 