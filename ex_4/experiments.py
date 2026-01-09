"""
Backwards-compatible re-exports from mlp_experiments and image_experiments.
"""
# MLP experiments (Section 6)
from mlp_experiments import (
    Q_1,
    Q2_Experiment,
    Q2_Results,
    Q4_Experiment,
    Q4_Results,
    Q6_2_Experiment,
    Q6_2_5_Experiment,
    Q6_2_5_Results,
    Q6_2_Results,
)

# Image/CNN experiments (Section 7)
from image_experiments import (
    CNNTrainer,
    BaseExperiment,
    ScratchExperiment,
    LinearProbeExperiment,
    FineTuneExperiment,
    XGBExperiment,
    SklearnProbeExperiment,
    Section7Aggregator,
    Section7Results,
)

# Re-export core classes
from MLP import MLP, MLPTrainer, EuropeDataset
