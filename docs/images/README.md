# FinSentinel Workflow Diagrams

This directory contains workflow diagrams for the FinSentinel project. These diagrams illustrate the architecture, data flows, and processing pipelines of the system.

## Available Diagrams

- `overall_workflow.png` - The complete workflow from data collection to visualization
- `system_architecture.png` - The architecture of the FinSentinel system
- `data_pipeline.png` - The data collection and processing pipeline
- `sentiment_analysis.png` - The sentiment analysis process using LLMs
- `backtest_workflow.png` - The backtesting and strategy development workflow
- `user_workflow.png` - The typical user interaction sequence

## Source Files

Each diagram is available in two formats:
- `.png` - Ready-to-use image file
- `.mmd` - Mermaid source file that can be edited

## Usage

To include these diagrams in documentation:

```markdown
![Overall Workflow](images/overall_workflow.png)
```

To include in Jupyter notebooks:

```python
from IPython.display import Image
Image(filename='../docs/images/overall_workflow.png')
```

## Modifying Diagrams

To modify the diagrams:

1. Edit the `.mmd` source files
2. Generate new PNG images using the conversion script:
   ```bash
   python convert_diagrams.py
   ```

## Requirements for Generating Diagrams

To generate PNG images from the Mermaid source files, you'll need:

1. Node.js installed
2. Mermaid CLI installed:
   ```bash
   npm install -g @mermaid-js/mermaid-cli
   ``` 