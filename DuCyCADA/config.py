"""
config.py
---------
Centralized configuration for DuCyCADA training and evaluation.

All hyper-parameters, dataset paths, and file-name patterns live here so
that train.py and evaluate.py never contain hard-coded magic numbers.

Edit the values in this file to adapt DuCyCADA to a new dataset or
experimental setup; no changes to the training / evaluation scripts are
required for most scenarios.
"""

# ---------------------------------------------------------------------------
# Dataset paths and glob patterns
# ---------------------------------------------------------------------------

#: Root directory that contains all image sub-folders.
DATASET_PATH = "archive1/"

#: Glob pattern (relative to DATASET_PATH) for source HR training images (Y_s).
TRAIN_HR_GLOB = "Dist rad train img_new/*.*"

#: Glob pattern for source LR training images (X_s, simulated motion-corrupted).
TRAIN_LR_GLOB = "Dist rad train img_new simz/*.*"

#: Glob pattern for unpaired target domain images (X_t).
TARGET_GLOB = "Dist rad target img/*.*"

#: Glob pattern for test LR images.
TEST_LR_GLOB = "Test Dataset Domain Adaptation/DA_dist_tib/img sim/*.*"

#: Glob pattern for test HR reference images.
TEST_HR_GLOB = "Test Dataset Domain Adaptation/DA_dist_tib/img/*.*"

# ---------------------------------------------------------------------------
# Image dimensions
# ---------------------------------------------------------------------------

#: Height of images fed to the network (pixels).
HR_HEIGHT = 256

#: Width of images fed to the network (pixels).
HR_WIDTH = 256

#: Number of image channels (1 = grayscale).
CHANNELS = 1

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

#: Mini-batch size. Set to 1 for memory-efficient single-sample training.
BATCH_SIZE = 1

#: Total number of training iterations (epochs in the random-sample loop).
#: Default: 15 120 source samples × 50 passes ≈ 756 000.
N_EPOCHS = 15120 * 50

#: Adam learning rate.
LR = 0.00008

#: Adam β₁ (first-moment decay).
B1 = 0.5

#: Adam β₂ (second-moment decay).
B2 = 0.999

#: Adam ε (numerical stability).
EPS = 1e-8

#: Epoch at which to begin learning-rate decay (used by schedulers).
DECAY_EPOCH = 100

# ---------------------------------------------------------------------------
# Dataset split sizes
# ---------------------------------------------------------------------------

#: Number of samples in the pre-loaded source split (index upper bound − 1).
N_SOURCE_SAMPLES = 15120   # indices 0 … 15119

#: Number of samples in the pre-loaded target split (index upper bound − 1).
N_TARGET_SAMPLES = 6720    # indices 0 … 6719

# ---------------------------------------------------------------------------
# Checkpoint and output paths
# ---------------------------------------------------------------------------

#: Path to the pre-loaded GPU dataset (created by the pre-loading step).
STACKED_DATASET_PATH = "stacked_dataset_DA_Dist_rad.pt"

#: Save a full training checkpoint every this many epochs.
SAVE_INTERVAL = 15000

#: Path where the best (lowest G_loss) generator weights are written.
BEST_MODEL_PATH = "saved_models/DuCyCADA_best_G_forward.pth"

#: Directory for evaluation result images.
RESULTS_DIR = "Results/DA_Dist_tib_results/"

# ---------------------------------------------------------------------------
# Number of CPU workers for DataLoader
# ---------------------------------------------------------------------------
N_CPU = 8
