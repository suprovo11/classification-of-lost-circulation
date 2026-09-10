### **Lost Circulation Classification Using Machine Learning Algorithms**



### **Overview**

This project develops a machine-learning framework for **classifying lost-circulation severity during drilling operations**. The workflow uses drilling/well data to distinguish between different levels of circulation loss and evaluates both tree-based and neural-network models.

The classification framework considers five circulation-loss categories:

* **No Loss**
* **Seepage**
* **Partial**
* **Severe**
* **Complete**



The project also investigates whether incorporating **depth-window/sequence-aware features** improves classification performance compared with using raw features alone.



### **Objectives**

The main objectives of this project are to:

1. Develop a machine-learning workflow for lost-circulation classification.
2. Classify drilling conditions into five levels of circulation-loss severity.
3. Compare the performance of **Gradient Boosted Decision Trees (GBDT)** and neural-network models.
4. Evaluate the contribution of depth-window/sequence-aware features.
5. Investigate model performance under a more stringent **well-grouped train/test split**.
6. Assess whether high classification performance generalizes across different wells.



### **Machine Learning Models**

The current workflow evaluates four model configurations:

|**Model**|**Feature Configuration**|
|-|-|
|**GBDT**|Raw features + balanced training|
|**GBDT + Depth Window**|Sequence-aware depth-window features|
|**Compact MLP**|Sequence features|
|**Paper-style NN Replica**|Raw features|

The notebook reports **17 raw features** and **43 features after incorporating window-based features**.



### **Methodology**

The general workflow is:

```text
Well / Drilling Data
        │
        ▼
Data Preparation
        │
        ▼
Well Segment Derivation
        │
        ▼
Anomaly Identification
        │
        ▼
Feature Engineering
        │
        ├── Raw Features
        │
        └── Depth-Window / Sequence Features
        │
        ▼
Machine Learning Models
        │
        ├── GBDT
        ├── GBDT + Window Features
        ├── Compact MLP
        └── Neural Network Replica
        │
        ▼
Model Evaluation
        │
        ├── Accuracy
        ├── Macro-F1
        ├── Weighted-F1
        ├── Precision
        └── Recall
        │
        ▼
Well-Grouped Generalization Analysis




The current pipeline derived **22 well segments** and flagged **6 anomalous rows** during processing.



### **Preliminary Results**

#### **Standard Evaluation**

The strongest preliminary result was obtained using **GBDT with depth-window/sequence-aware features**:

|Model|Accuracy|Macro-F1|Weighted-F1|
|-|-:|-:|-:|
|GBDT — Raw|0.9900|0.9493|0.9900|
|**GBDT + Depth Window**|**0.9918**|**0.9649**|**0.9918**|
|Compact MLP|0.9832|0.9318|0.9832|
|Paper-style NN Replica|0.9627|0.8858|0.9627|

The GBDT + depth-window configuration achieved the highest reported accuracy and Macro-F1 among the evaluated models.



#### **Well-Grouped Evaluation**

A more stringent **well-grouped split** produced substantially lower performance:

* **Accuracy:** 0.6849
* **Macro-F1:** 0.2009

In this evaluation, the model performed strongly on the majority **No Loss** class but failed to reliably identify the rarer **Partial, Severe, and Complete** categories.

This difference is an important finding because it indicates that the very high standard-split performance should **not automatically be interpreted as strong cross-well generalization**.



#### **Evaluation Metrics**

Model performance is evaluated using:

* **Accuracy**
* **Macro-F1 Score**
* **Weighted-F1 Score**
* **Precision**
* **Recall**
* Class-level F1 scores

Macro-F1 is particularly important for this project because the dataset contains substantial class imbalance, especially for the **Severe** and **Complete** circulation-loss categories. The reported evaluation contains only 68 Severe and 8 Complete samples in the standard test results.



### **Installation**

Clone the repository:

```bash
git clone https://github.com/suprovo11/classification-of-lost-circulation
cd lost-circulation-classification
```

Create a Python environment:

```bash
python -m venv .venv
```

Activate the environment:

### **Windows**

```bash
.venv\\Scripts\\activate
```

### **Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

\---

### **Running the Project**

The main pipeline can be executed with:

```bash
python pipeline.py
```

The notebook uses the same pipeline command:


!python pipeline.py


The current workflow generates a `results.json` file containing the model evaluation results.

\---

### **Notebook**

The repository includes a Jupyter Notebook for running and reviewing the classification workflow:

```text
notebooks/classify\_lost\_circulation.ipynb
```

The notebook is intended to provide a reproducible entry point for inspecting model results and the machine-learning workflow.

\---

## **Key Findin**g

The preliminary results demonstrate an important distinction between **within-dataset predictive performance** and **cross-well generalization**.

Although the standard evaluation produced approximately **99% accuracy**, the well-grouped evaluation reduced accuracy to approximately **68.5%**, with a Macro-F1 of only **0.20**.
Therefore, future development should focus on:

* Better cross-well validation
* Improved handling of minority classes
* Prevention of data leakage
* Class-balanced evaluation
* More robust feature engineering
* Testing on completely unseen wells
* Investigation of false predictions for Severe and Complete losses



### **Current Limitations**

The current results reveal several areas requiring further investigation:

1. **Class imbalance** — Severe and Complete losses have very few samples.
2. **Generalization** — performance decreases substantially under well-grouped evaluation.
3. **Minority-class detection** — Partial, Severe, and Complete classes require further improvement.
4. **Potential dataset dependence** — very high standard-split performance should be interpreted carefully.
5. **Cross-well robustness** — unseen-well testing should be prioritized before claiming operational deployment.

### 

### **Future Work**

Future versions of the project may include:

* Cross-validation by well
* Leave-one-well-out validation
* SMOTE or other imbalance-handling strategies
* XGBoost / LightGBM comparison
* Random Forest benchmarking
* CNN/LSTM/GRU sequence models
* Explainable AI using SHAP
* Feature-importance analysis
* Confusion-matrix visualization
* ROC-AUC and Precision-Recall analysis
* Hyperparameter optimization
* Real-time lost-circulation prediction
* Deployment as a drilling decision-support system



### **Research Relevance**

This project is relevant to **drilling engineering, petroleum engineering, geomechanics, and machine learning**. Automated classification of circulation-loss severity can potentially support earlier identification of abnormal drilling conditions and contribute to data-driven drilling decision support.



### **Author**

**Salman Shakib Suprova**   
Department of Petroleum and Mining Engineering  
Shahjalal University of Science and Technology (SUST), Bangladesh



## L**icense**

This project is intended for academic and research purposes. 

