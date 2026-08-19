# MLOps Weekly Assingments
**Week 9 Graded Assingment**

## Model Card: IRIS Classifier

### Model Details
- **Model Type:** Decision Tree Classifier
- **Framework:** scikit-learn
- **Training Features:** sepal_length, sepal_width, petal_length, petal_width

### Training Data
- IRIS dataset (150 samples, 4 features)
- 80/20 train/test split

### Performance Metrics
| Metric | Overall |
|--------|---------|
| Accuracy | ~99% |
| Precision | ~99% |
| Recall | ~99% |

### Fairness Analysis
- Sensitive attribute: `location` (randomly assigned, binary)
- Metrics evaluated per group using Fairlearn MetricFrame
- Expected near-equal performance across groups (random assignment)

### Known Limitations
- Small dataset (150 samples)
- Location attribute is synthetic, not real-world
- May not generalize to other flower datasets

### Drift Monitoring
- Simulated drift by shifting petal_length (+1.5) and petal_width (+0.5)
- Drift detected using Evidently AI DataDriftPreset