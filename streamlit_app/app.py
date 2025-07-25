import streamlit as st
import sys
import os
import json
import tempfile
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from task3_extraction.schema import BusinessRule, ExtractedKnowledge
    SCHEMA_AVAILABLE = True
except ImportError:
    SCHEMA_AVAILABLE = False
    st.error("Schema module not found. Please run from project root directory.")

import networkx as nx

# Page config
st.set_page_config(
    page_title="Circonomit - Language to Structure Demo", 
    page_icon="🧠",
    layout="wide"
)

# Header
st.title("Circonomit - Language to Structure Demo")
st.markdown("**Transform natural language business rules into structured Block-Attribute relationships**")

# Two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Natural Language Input")
    
    # Sample rules for quick testing
    sample_rules = {
        "Energy Price Rule": "If energy costs rise above €200/MWh, we will postpone production by one week.",
        "Demand Shift Rule": "Demand in France drops by 60% at a price above €250 per unit.",
        "Customer Loss Rule": "If we can't deliver for two weeks, there's an 80% chance we'll lose the customer.",
        "Custom Rule": ""
    }
    
    selected_rule = st.selectbox("Choose a sample rule or enter custom:", list(sample_rules.keys()))
    
    if selected_rule == "Custom Rule":
        rule_text = st.text_area(
            "Enter your business rule:",
            height=120,
            placeholder="e.g., 'If inventory drops below 100 units, increase production by 20%'"
        )
    else:
        rule_text = st.text_area(
            "Business rule:",
            value=sample_rules[selected_rule],
            height=120
        )
    
    extract_button = st.button("Extract Knowledge", type="primary", use_container_width=True)

with col2:
    st.subheader("Structured Output")
    
    if extract_button and rule_text.strip():
        if not SCHEMA_AVAILABLE:
            st.error("Schema not available. Please check module imports.")
        else:
            try:
                # Mock extraction for demo (replace with real LLM in production)
                def mock_extract_rule(text: str) -> BusinessRule:
                    """Mock extraction that simulates LLM response"""
                    
                    # Pre-defined responses for demo
                    mock_responses = {
                        "If energy costs rise above €200/MWh, we will postpone production by one week.": {
                            "condition_block": "Energy",
                            "condition_attribute": "price_future",
                            "comparator": ">",
                            "threshold": 200.0,
                            "threshold_unit": "€/MWh",
                            "action_block": "Production",
                            "action_attribute": "schedule_delay",
                            "action_value": "7 days",
                            "action_type": "delay",
                            "confidence": 0.9,
                            "temporal_delay": "1 week",
                            "probability": None,
                            "original_text": text
                        },
                        "Demand in France drops by 60% at a price above €250 per unit.": {
                            "condition_block": "Production",
                            "condition_attribute": "unit_cost",
                            "comparator": ">",
                            "threshold": 250.0,
                            "threshold_unit": "€/unit",
                            "action_block": "Demand",
                            "action_attribute": "demand_shift",
                            "action_value": "-0.6",
                            "action_type": "decrease",
                            "confidence": 0.85,
                            "temporal_delay": None,
                            "probability": None,
                            "original_text": text
                        },
                        "If we can't deliver for two weeks, there's an 80% chance we'll lose the customer.": {
                            "condition_block": "Inventory",
                            "condition_attribute": "backlog",
                            "comparator": ">=",
                            "threshold": 14.0,
                            "threshold_unit": "days",
                            "action_block": "Demand",
                            "action_attribute": "customer_retention",
                            "action_value": "0.2",
                            "action_type": "probability",
                            "confidence": 0.8,
                            "temporal_delay": "2 weeks",
                            "probability": 0.8,
                            "original_text": text
                        }
                    }
                    
                    # Use exact match or create a generic response
                    if text in mock_responses:
                        data = mock_responses[text]
                    else:
                        # Generic extraction for custom rules
                        data = {
                            "condition_block": "Energy",
                            "condition_attribute": "price_now",
                            "comparator": ">",
                            "threshold": 100.0,
                            "threshold_unit": "€",
                            "action_block": "Production",
                            "action_attribute": "adjustment",
                            "action_value": "custom_action",
                            "action_type": "set",
                            "confidence": 0.6,
                            "temporal_delay": None,
                            "probability": None,
                            "original_text": text
                        }
                    
                    return BusinessRule(**data)
                
                # Extract the rule
                with st.spinner("Extracting knowledge..."):
                    rule_obj = mock_extract_rule(rule_text)
                
                # Success metrics
                st.success(f"Extraction Complete (Confidence: {rule_obj.confidence:.1%})")
                
                # Show structured JSON
                st.markdown("**Extracted JSON:**")
                st.json(rule_obj.model_dump(), expanded=True)
                
                # Graph mapping visualization
                st.markdown("**Graph Relationship:**")
                condition_node = f"{rule_obj.condition_block}.{rule_obj.condition_attribute}"
                action_node = f"{rule_obj.action_block}.{rule_obj.action_attribute}"
                
                # Create simple graph visualization
                G = nx.DiGraph()
                G.add_edge(condition_node, action_node, 
                          label=f"{rule_obj.comparator} {rule_obj.threshold} {rule_obj.threshold_unit or ''}")
                
                # Simple text representation
                st.code(f"""
CONDITION: {condition_node}
TRIGGER:   {rule_obj.comparator} {rule_obj.threshold} {rule_obj.threshold_unit or ''}
ACTION:    {action_node} = {rule_obj.action_value}
TYPE:      {rule_obj.action_type}
                """)
                
                # Integration preview
                st.markdown("**Integration with Task 1:**")
                st.info(f"""
                This rule extends the Wirkungsnetz with:
                • New conditional edge: {condition_node} → {action_node}
                • Threshold monitoring: {rule_obj.threshold} {rule_obj.threshold_unit or ''}
                • Action type: {rule_obj.action_type}
                • Ready for simulation injection
                """)
                
            except Exception as e:
                st.error(f"❌ Extraction failed: {str(e)}")
                st.exception(e)
    
    elif extract_button and not rule_text.strip():
        st.warning("⚠️ Please enter a business rule to extract")

# Sidebar with information
with st.sidebar:
    st.markdown("### About This Demo")
    st.markdown("""
    This interactive demo shows how Circonomit converts natural language business rules into structured Block-Attribute relationships.
    
    **Pipeline:**
    1. **NLP Extraction** - Parse rule patterns
    2. **Schema Validation** - Ensure data consistency  
    3. **Graph Mapping** - Create Block-Attribute edges
    4. **Simulation Ready** - Inject into Task 1 model
    """)
    
    st.markdown("### Technical Stack")
    st.markdown("""
    • **Frontend**: Streamlit
    • **Schema**: Pydantic validation
    • **LLM**: GPT-4 (production)
    • **Graph**: NetworkX integration
    • **Deployment**: Streamlit Cloud
    """)
    
    st.markdown("### Task Integration")
    st.markdown("""
    • **Task 1**: Extends simulation graph
    • **Task 3**: Feeds rule engine
    • **Task 4**: Powers expert interface
    """)

# Footer
st.markdown("---")
st.markdown("**Circonomit Challenge - Task 3: Language to Structure Knowledge Extraction**")
st.markdown("*Transforming expert knowledge into computational models*") 