::: content
[]{#markdown-mermaid aria-hidden="true" dark-mode-theme="dark"
light-mode-theme="default" max-text-size="50000"}

# Smart Test Predictor Design & Implementation {#smart-test-predictor-design--implementation .code-line line="0" dir="auto"}

## Project Overview {#project-overview .code-line line="2" dir="auto"}

This project develops a Smart Test Predictor for the semiconductor
industry to predict chip malfunction using early testing data. The
system predicts whether a chip will pass the Final Test (FT) flow based
on measurement data collected during the Chip Probing (CP) flow.

Chips undergo two main test flows:

- **CP (Chip Probing) flow**: Early testing phase that generates
  measurement data
- **FT (Final Test) flow**: Final testing phase that determines chip
  viability

Each flow consists of multiple test groups, and each test group contains
multiple test conditions with specific measurement parameters.
Generally, it can be expected that a parameter name in a given test
group will be present in all test groups within that same flow.

## Current Testing Architecture {#current-testing-architecture .code-line line="13" dir="auto"}

### CP Flow Test Groups {#cp-flow-test-groups .code-line line="15" dir="auto"}

1.  **CP1** (room temperature)
2.  **CP1.5** (hot temperature)
3.  **YPP** (hot temperature)
4.  **QPP** (cold temperature)
5.  **RPP** (room temperature)
6.  **JPP** (hot temperature)

### FT Flow Test Groups {#ft-flow-test-groups .code-line line="24" dir="auto"}

1.  **FT1** (hot temperature)
2.  **FT2** (cold temperature)
3.  **FT3** (room temperature)

## System Requirements {#system-requirements .code-line line="30" dir="auto"}

1.  **Limited FT Flow Visibility**: The system must only receive a
    single boolean outcome from the FT flow indicating whether the chip
    passed all tests. Internal FT flow details (e.g., rule-based
    criteria, sensor offset calculations) must remain hidden from the
    Smart Test Predictor.

2.  **Flexible Data Handling**: The application must adjust to different
    columns being available for each inference request. This is to
    accommodate a dynamic test environment where some test groups may be
    skipped or modified, leading to varying measurement parameters.

## Solution Architecture {#solution-architecture .code-line line="36" dir="auto"}

The ML engine implements a schema-adaptive \"just-in-time\" (JIT)
learning approach to accommodate dynamic test conditions:

### Inference Process Introduction {#inference-process-introduction .code-line line="40" dir="auto"}

1.  **Column Selection & Filtering**: Analyze and filter the columns in
    the inference dataset through multiple criteria including coverage
    assessment, predictive power analysis, and user overrides.

2.  **Historical Data Discovery**: Load historical data with exact or
    superset column matches. Records missing required columns are
    rejected.

3.  **Model Training**: Train a model using the matched historical data
    with the final column set

4.  **Prediction**: Apply the trained model to predict FT flow outcomes
    using only the columns available in both datasets

This architecture ensures the system remains flexible and can adapt to
evolving test configurations without manual intervention.

### Inference Process Flowchart {#inference-process-flowchart .code-line line="52" dir="auto"}

```{style="all:unset;"}
---
config:
  theme: 'base'
  themeVariables:
    background: '#f8f9fa'
    primaryColor: '#ffffff'
    primaryTextColor: '#000000'
    primaryBorderColor: '#7C7C7C'
    lineColor: '#7C7C7C'
    secondaryColor: '#ffffff'
    tertiaryColor: '#ffffff'
---
flowchart TD
    A[New Inference Request] --> B[1: Column Selection & Filtering]
    B --> C[2: Historical Data Discovery]
    C --> D{Sufficient Compatible<br/>Historical Data?}
    
    D -->|No| E[Return Error:<br/>Insufficient Training Data]
    D -->|Yes| F[3: Model Training<br/>Dynamic Algorithm Selection]
    
    F --> G{Model Performance<br/>Acceptable?}
    G -->|No| H[Adjust Parameters]
    H --> F
    
    G -->|Yes| I[4: Prediction<br/>Generate Pass/Fail Results]
    I --> J[Return Predictions]
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000000
    style J fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#000000
    style E fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000000
    style D fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000000
    style G fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000000
```

## Process Details {#process-details .code-line line="89" dir="auto"}

### 1. Column Selection & Filtering Process {#1-column-selection--filtering-process .code-line line="91" dir="auto"}

The system analyzes and filters the columns in the inference dataset
through three criteria:

- **Column Coverage Assessment**: Exclude columns that are not available
  or have limited data (below threshold) in historical records
- **Automated Predictive Power Assessment**: Exclude columns known to
  have poor predictive power through automated analysis
- **User Override**: Allow users final authority to include or exclude
  any columns at their discretion

### 2. Historical Data Discovery Process {#2-historical-data-discovery-process .code-line line="99" dir="auto"}

When an inference dataset is loaded into the application, the system
initiates a discovery process to identify compatible historical training
data. This process provides transparency to users about data
availability and matching quality.

#### Matching Strategy {#matching-strategy .code-line line="103" dir="auto"}

The system identifies compatible historical data using column matching:

- **Exact Match**: Historical records with identical column sets
- **Superset Match**: Historical records containing all inference
  columns plus additional ones (additional columns are ignored during
  training)

**Matching Process**:

1.  Find all records with exact or superset column matches
2.  Use all compatible records for training (combining exact and
    superset matches)
3.  Reject records missing any required inference columns

**Example**:

- Inference columns: `[alpha, beta, charlie]`
- Historical record A: `[alpha, beta, charlie]` → Exact match (use all)
- Historical record B: `[alpha, beta, charlie, delta]` → Superset
  (ignore delta)
- Historical record C: `[alpha, beta]` → Rejected (missing required
  columns)

#### Discovery Output Example {#discovery-output-example .code-line line="123" dir="auto"}

``` {.code-line .language-text line="125" dir="auto"}
Inference Dataset Analysis:
- Columns detected: [CP1_VOLTAGE, CP1_CURRENT, YPP_TEMP, QPP_OFFSET]
- Total inference records: 150 chips

Historical Data Discovery Results:
✓ Compatible data:   3,350 chips (exact + superset matches)
  - Exact matches:     1,250 chips (identical column sets)
  - Superset matches:  2,100 chips (contain additional columns)
✗ Incompatible data: 1,170 chips (missing required columns - excluded)

Training Strategy:
- Training dataset: 3,350 chips (all compatible historical data)
- Model features: 4 columns (all inference columns available)
```

The application provides transparency about data availability and
automatically uses all compatible historical data for optimal model
training.

### 3. Model Training Process {#3-model-training-process .code-line line="143" dir="auto"}

*\[Placeholder for model training implementation\]*

### 4. Prediction Process {#4-prediction-process .code-line line="147" dir="auto"}

*\[Placeholder for prediction implementation\]*
:::
