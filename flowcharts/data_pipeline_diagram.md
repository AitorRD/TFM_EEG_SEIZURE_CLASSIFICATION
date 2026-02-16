# EEG Data Processing Pipeline

## Step 1: Conversion and Preprocessing

```mermaid
flowchart LR
    Input[(EDF Files<br/>Siena Dataset)] --> Filter[Filtering<br/>0.5-70 Hz]
    Filter --> ICA[ICA<br/>Artifacts]
    ICA --> Resample[Resampling<br/>100 Hz]
    Resample --> Label[Seizure<br/>Labeling]
    Label --> Clip[Clipping<br/>±30 min]
    Clip --> Output[(CSV Files<br/>~40 files)]
    
    classDef process fill:#9ca3af,stroke:#6b7280,stroke-width:1.5px,color:#000
    classDef data fill:#f3f4f6,stroke:#9ca3af,stroke-width:1.5px,color:#000
    class Filter,ICA,Resample,Label,Clip process
    class Input,Output data
```

**Input:** Raw EDF files from Siena dataset  
**Processes:** Filtering → ICA → Resampling → Labeling → Temporal clipping  
**Output:** ~40 preprocessed CSVs in `data/raw/csv-data/`

---

## Step 2: Dataset Split

```mermaid
flowchart TD
    Input[(40 CSV files)] --> Group[Group by<br/>Patient]
    Group --> Split{Split<br/>70/15/15}
    Split --> Train[(train.csv)]
    Split --> Val[(val.csv)]
    Split --> Test[(test.csv)]
    
    classDef process fill:#9ca3af,stroke:#6b7280,stroke-width:1.5px,color:#000
    classDef data fill:#f3f4f6,stroke:#9ca3af,stroke-width:1.5px,color:#000
    classDef split fill:#9ca3af,stroke:#6b7280,stroke-width:1.5px,color:#000
    class Group process
    class Input,Train,Val,Test data
    class Split split
```

**Input:** Individual CSV files per session  
**Processes:** Patient-level grouping → Stratified split (no data leakage)  
**Output:** 3 consolidated files in `data/processed/dataset_clipped/`

---

## Step 3: Window Generation

```mermaid
flowchart LR
    Input[(train.csv<br/>val.csv<br/>test.csv)] --> Window[Windowing 10s<br/>Overlap 25%]
    Window --> Train[(windowed_train.csv)]
    Window --> Val[(windowed_val.csv)]
    Window --> Test[(windowed_test.csv)]
    
    classDef process fill:#9ca3af,stroke:#6b7280,stroke-width:1.5px,color:#000
    classDef data fill:#f3f4f6,stroke:#9ca3af,stroke-width:1.5px,color:#000
    class Window process
    class Input,Train,Val,Test data
```

**Input:** Concatenated datasets (train/val/test)  
**Processes:** Sliding windows of 10s with 25% overlap  
**Output:** Windowed datasets ready for ML/DL in `data/processed/windowed/`

---

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| EEG Channels | 22 (10-20 system) |
| Sampling Rate | 100 Hz |
| Window Size | 10s (1000 samples) |
| Data Reduction | ~60-80% after clipping |

## Usage

```bash
python data.py                   # Complete pipeline
python data.py --step conversion # Step 1 only
python data.py --step concat     # Step 2 only
python data.py --step window     # Step 3 only
```
