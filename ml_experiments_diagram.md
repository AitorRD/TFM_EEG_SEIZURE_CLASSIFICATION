# ML Experiments Flow Diagram

```mermaid
flowchart TD
    %% Input data with parallel splits
    Windowed[(Windowed Data)] 
    
    Windowed --> Train[Train]
    Windowed --> Val[Val]
    Windowed --> Test[Test]
    
    %% Feature Extraction
    Train --> FETrain[Feature<br/>Extraction]
    Val --> FEVal[Feature<br/>Extraction]
    Test --> FETest[Feature<br/>Extraction]
    
    FETrain --> FS[Feature<br/>Selection]
    FS --> TRTrain[Selection<br/>Transform]
    FEVal --> TRVal[Transform]
    FETest --> TRTest[Transform]
    
    %% Pipeline train
    TRTrain --> CV[Cross<br/>Validation]
    CV --> TM[Train<br/>Models]
    TM --> EV[Eval<br/>Validation]
    EV --> ET[Eval<br/>Test]
    ET --> XAI[XAI<br/>Explanations]

    %% Pipeline val
    TRVal --> SB[Select Best<br/>]

    %% Pipeline test
    TRTest --> SBTest[Eval<br/>Test]

    classDef process fill:#9ca3af,stroke:#6b7280,stroke-width:1.5px,color:#000
    classDef data fill:#f3f4f6,stroke:#9ca3af,stroke-width:1.5px,color:#000
    
    class Train,Val,Test,FETrain,FEVal,FETest,FS,CV,TM,EV,ET,XAI process
    class Windowed data
```

**Pipeline Steps:**
1. **Split** windowed data → train/val/test (70/15/15)
2. **Feature extraction** TSFRESH ~800 features, fit selector on train
3. **Feature selection** SelectKBest fit on train, transform all sets
4. **Cross validation** StratifiedKFold + hyperparameter tuning on train
5. **Train models** LR, RF, SVC, KNN, XGBoost with best params
6. **Eval validation** train + val → select best model
7. **Eval test** best model + test → final metrics + XAI (SHAP, LIME)

---

## Technical Specifications

| Component | Details |
|-----------|---------|
| **ML Algorithms** | Logistic Regression, Random Forest, SVC, KNN, XGBoost |
| **Feature Extraction** | TSFRESH (~800 time-series features) |
| **Feature Selection** | SelectKBest with ANOVA F-test |
| **Cross Validation** | StratifiedKFold (5 folds) |
| **Optimization** | Optuna with TPE sampler |
| **XAI Methods** | SHAP global importance, LIME local explanations |
| **Metrics** | Accuracy, Precision, Recall, F1-Score, ROC-AUC |

## Usage

```bash
cd experimentation/classic
python ml_experiments.py
```

**Configuration:** `experimentation/classic/config.yaml`
