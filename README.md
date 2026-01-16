# PDPE: Patient Data Privacy & Encryption

🛡️ **PDPE** is an advanced medical data analysis platform designed to detect anomalies and privacy risks using a multi-layered approach: Rule-based validation, Machine Learning (Isolation Forest), and Deep Learning (Autoencoders).

## Core Features
- **Multi-Layer Detection**: Combines expert rules with AI-driven anomaly detection.
- **Risk Scoring**: Aggregates findings into a single, intuitive Risk Percentage and Category.
- **Interactive Dashboards**: Visualizes risk distributions and detailed record breakdowns using Streamlit and Plotly.
- **Data Validation**: Ensures data integrity for critical medical fields.

## Project Structure
```text
PDPE_Streamlit_App/
├── app.py                     # Main Streamlit app
├── requirements.txt           # Dependencies
├── models/                    # Trained ML/DL models
├── core/
│   ├── rules.py               # Rule-based logic
│   ├── ml.py                  # ML Anomaly detection
│   ├── dl.py                  # DL Anomaly detection
│   └── risk.py                # Risk aggregation
├── utils/
│   └── data_utils.py          # Data helpers
└── data/                      # Sample datasets
```

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Sample Data
A sample dataset `medical_data_mixed_values.csv` is provided in the `data/` directory for testing. It contains intentionally erroneous data (e.g., age 150, invalid blood types) to demonstrate detection capabilities.
