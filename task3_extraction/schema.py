from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

class BusinessRule(BaseModel):
    """
    Schema for extracting business rules from natural language.
    Maps to Circonomit Block-Attribute model for graph representation.
    """
    
    # Condition (trigger)
    condition_block: str = Field(..., description="Source block name (Energy, Production, Demand, Inventory)")
    condition_attribute: str = Field(..., description="Attribute being monitored (price_now, stock, etc.)")
    comparator: Literal[">", ">=", "<", "<=", "==", "!="] = Field(..., description="Comparison operator")
    threshold: float = Field(..., description="Threshold value that triggers the rule")
    threshold_unit: Optional[str] = Field(None, description="Unit of measurement (€/MWh, days, %, etc.)")
    
    # Action (consequence)
    action_block: str = Field(..., description="Target block for the action")
    action_attribute: str = Field(..., description="Attribute to modify or create")
    action_value: str = Field(..., description="New value or modification (can include units)")
    action_type: Literal["set", "increase", "decrease", "delay", "probability"] = Field(..., description="Type of action to perform")
    
    # Metadata
    confidence: Optional[float] = Field(0.8, description="Extraction confidence score (0-1)")
    temporal_delay: Optional[str] = Field(None, description="Time delay for the action (1 week, 2 days, etc.)")
    probability: Optional[float] = Field(None, description="Probability of action occurring (0-1)")
    
    # Source tracking
    original_text: str = Field(..., description="Original natural language rule")
    extracted_at: datetime = Field(default_factory=datetime.now, description="When this rule was extracted")
    
    class Config:
        schema_extra = {
            "example": {
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
                "original_text": "If energy costs rise above €200/MWh, we will postpone production by one week."
            }
        }

class ExtractedKnowledge(BaseModel):
    """
    Container for multiple extracted rules with validation metadata.
    """
    rules: list[BusinessRule] = Field(..., description="List of extracted business rules")
    extraction_method: str = Field(..., description="Method used for extraction (llm, rule-based, hybrid)")
    validation_status: Literal["pending", "validated", "rejected"] = Field("pending", description="Human validation status")
    validation_notes: Optional[str] = Field(None, description="Human reviewer notes")
    
    def to_graph_format(self) -> list[dict]:
        """
        Convert extracted rules to graph database format.
        Returns list of condition-action relationships.
        """
        graph_rules = []
        for rule in self.rules:
            graph_rule = {
                "condition": {
                    "block": rule.condition_block,
                    "attribute": rule.condition_attribute,
                    "operator": rule.comparator,
                    "threshold": rule.threshold,
                    "unit": rule.threshold_unit
                },
                "action": {
                    "block": rule.action_block,
                    "attribute": rule.action_attribute,
                    "value": rule.action_value,
                    "type": rule.action_type,
                    "delay": rule.temporal_delay,
                    "probability": rule.probability
                },
                "metadata": {
                    "confidence": rule.confidence,
                    "source": rule.original_text
                }
            }
            graph_rules.append(graph_rule)
        return graph_rules

# Validation functions
def validate_block_names(rule: BusinessRule) -> bool:
    """Validate that block names match Task 1 model."""
    valid_blocks = {"Energy", "Production", "Demand", "Inventory"}
    return (rule.condition_block in valid_blocks and 
            rule.action_block in valid_blocks)

def validate_attributes(rule: BusinessRule) -> bool:
    """Validate that attributes exist in the respective blocks."""
    block_attributes = {
        "Energy": ["price_now", "tariff_future", "price_future", "CO2_factor"],
        "Production": ["base_cost", "unit_cost", "unit_emissions"],
        "Demand": ["demand_today", "demand_shift"],
        "Inventory": ["stock", "backlog"]
    }
    
    condition_valid = rule.condition_attribute in block_attributes.get(rule.condition_block, [])
    # Allow new attributes for actions (they might create new calculated fields)
    return condition_valid 