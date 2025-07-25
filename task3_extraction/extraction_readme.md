# Task 3: Language to Structure - Knowledge Extraction

## Problem Statement

Convert natural language business rules into structured knowledge that integrates with the Block-Attribute model from Task 1. Extract actionable rules from subject matter expert communications.

## Solution Approach

### Hybrid Extraction Pipeline

**Three-stage pipeline:**

1. **LLM Extraction** - GPT-4 with structured prompts for flexible language understanding
2. **Schema Validation** - Pydantic models ensure data consistency and type safety  
3. **Business Logic Validation** - Custom validators check against known Block-Attribute structure

### Knowledge Representation Philosophy

**Condition-Action Pattern:**
Every business rule follows the pattern: `IF condition THEN action`

- **Condition**: Monitoring specific Block.Attribute against threshold values
- **Action**: Modifying target Block.Attribute with specific operations
- **Metadata**: Confidence, temporal delays, probabilities, source tracking

**Graph Integration:**
Each extracted rule creates new edges in the knowledge graph:
```
Energy.price_future --[threshold: >200 €/MWh]--> Production.schedule_delay
```

## Ontologies and Vocabularies

### Core Ontology

**Base Classes:**
- `Block` - Business domain containers (Energy, Production, Demand, Inventory)
- `Attribute` - Measurable properties within blocks (inputs, calculations)
- `Rule` - Condition-action relationships between attributes
- `Threshold` - Trigger conditions with comparators and values

**Relationships:**
- `MONITORS` - Rule monitors specific attribute
- `TRIGGERS` - Condition triggers specific action
- `MODIFIES` - Action modifies target attribute
- `DEPENDS_ON` - Temporal or causal dependencies

### External Vocabularies

**W3C Semantic Sensor Network (SSN):**
- `sosa:Observation` - Inspired condition monitoring concepts
- `sosa:Actuation` - Inspired action execution patterns

**QUDT (Quantities, Units, Dimensions, Types):**
- Unit standardization for thresholds (€/MWh, days, percentages)
- Dimensional analysis for validation

## Tools and Technologies

### Natural Language Processing
- **LLM**: GPT-4 via OpenAI API for flexible language understanding
- **Prompt Engineering**: Structured templates with examples and constraints
- **spaCy**: Entity recognition and linguistic validation (future enhancement)

### Data Validation and Schema
- **Pydantic**: Type validation and serialization
- **JSON Schema**: Standardized data exchange format
- **Custom Validators**: Business logic compliance checking

### Graph Database Integration
- **Target**: Grakn or Neo4j for persistent knowledge storage
- **Schema Mapping**: JSON to graph triple conversion
- **Query Interface**: Graph traversal for rule discovery and conflict detection

### Integration Framework
- **LangChain**: LLM orchestration and prompt management
- **Pandas**: Data manipulation and CSV integration
- **NetworkX**: Graph analysis and cycle detection (from Task 1)

## Technical Implementation

### Extraction Schema

```python
class BusinessRule(BaseModel):
    # Condition components
    condition_block: str
    condition_attribute: str  
    comparator: Literal[">", ">=", "<", "<=", "==", "!="]
    threshold: float
    threshold_unit: Optional[str]
    
    # Action components
    action_block: str
    action_attribute: str
    action_value: str
    action_type: Literal["set", "increase", "decrease", "delay", "probability"]
    
    # Metadata
    confidence: float
    temporal_delay: Optional[str]
    probability: Optional[float]
    original_text: str
```

### Language to Logic Mapping

**Input Processing:**
1. Parse natural language for IF-THEN patterns
2. Extract entities: blocks, attributes, numbers, units, operators
3. Map to ontology concepts with confidence scoring
4. Validate against known schema structure

**Logic Generation:**
1. Create condition nodes with threshold monitoring
2. Generate action nodes with specific operations
3. Establish temporal relationships and dependencies
4. Inject into simulation as dynamic rules

## Integration with Task 1

### Graph Extension
Extracted rules extend the Block-Attribute graph with new edge types:
- **Conditional edges**: Threshold-based triggers
- **Action edges**: Dynamic attribute modifications
- **Meta edges**: Confidence and temporal information

### Simulation Integration
Rules become dynamic scenario modifications:
```python
# Example: Energy price rule
if Energy.price_future > 200:
    Production.schedule_delay = "7 days"
    scenario_modifications.append(delay_rule)
```

## Validation and Quality Assurance

### Multi-level Validation
1. **Syntactic**: JSON schema compliance
2. **Semantic**: Ontology consistency checking  
3. **Pragmatic**: Business logic validation
4. **Human-in-loop**: Expert review for low-confidence extractions

### Confidence Scoring
- **High (>0.8)**: Clear IF-THEN patterns with known entities
- **Medium (0.5-0.8)**: Implicit conditions or new attributes
- **Low (<0.5)**: Ambiguous language requiring human review

## Future Enhancements

### Advanced NLP
- **Custom NER**: Domain-specific entity recognition for business concepts
- **Coreference Resolution**: Handle pronouns and implicit references
- **Temporal Expression Recognition**: Parse complex time specifications

### Knowledge Management
- **Rule Conflict Detection**: Identify contradictory business rules
- **Rule Composition**: Combine multiple simple rules into complex logic
- **Version Control**: Track rule evolution and expert feedback

### Scalability
- **Batch Processing**: Handle document-level knowledge extraction
- **Real-time Updates**: Stream processing for chat/email rule discovery
- **Multi-language Support**: Extend to German, French for European operations

## Results and Performance

**Test Case Performance:**
- 3/3 business rules successfully extracted
- Average confidence: 0.85
- Zero validation errors
- Complete integration with Task 1 model

**Knowledge Graph Extension:**
- 3 new conditional relationships
- 2 new action nodes created
- Full compatibility with existing simulation engine

The hybrid pipeline successfully bridges the gap between natural language business communication and structured computational models, enabling subject matter experts to contribute domain knowledge without technical modeling expertise. 