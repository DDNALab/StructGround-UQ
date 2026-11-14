\# StructGround-UQ — Week 3 Notes



\## Step 4: Environment Setup



Operating system: Windows



Environment name: structground-w3



Environment manager: Conda



Python version: 3.11



\### Hardware



GPU: NVIDIA RTX 2000 Ada Generation



GPU memory: approximately 16 GB



NVIDIA driver: 596.71



Maximum CUDA version supported by driver: 13.2



PyTorch version: 2.9.1+cu128



PyTorch CUDA runtime: 12.8



CUDA available: True



\### Software



Required packages:

\- PyTorch

\- torchvision

\- transformers

\- NumPy

\- SciPy

\- scikit-image



Exact versions are recorded in requirements.txt.



\### Verification



All required Python imports succeeded.



PyTorch successfully detected the NVIDIA GPU.



No model training or fine-tuning was performed.



The frozen train, calibration, and test splits

were not modified.





\## Week 3 — Step 11: Cost and Reproducibility



\### Hardware and environment

\- OS: Windows, build 26200

\- CPU: AMD64 Family 25 Model 24

\- GPU: NVIDIA RTX 2000 Ada Generation

\- GPU memory: 16 GB

\- PyTorch: 2.9.1+cu128

\- Transformers: 4.57.1

\- CUDA: Available

\- Random seed: 2026



\### Scoring runtimes

Original scoring runtimes were not recorded.



Measured reproducibility rerun times:

\- Calibration: 812.62 seconds (13 min 32.62 sec)

\- Test: 1012.72 seconds (16 min 52.72 sec)

\- Combined: 1825.34 seconds (30 min 25.34 sec)



Times include inference and SHA-256 verification.



\### Reproducibility verification

Calibration:

\- Images rescored: 1186

\- Matching hashes: 1186

\- Mismatches: 0

\- Result: PASS



Test:

\- Images rescored: 1187

\- Matching hashes: 1187

\- Mismatches: 0

\- Result: PASS



All regenerated score-file hashes matched the

original manifests exactly.



Original score caches were preserved.

Test performance metrics were not inspected.



Reproducibility summaries:

\- results/week3\_repro\_calib.json

\- results/week3\_repro\_test.json

## Week 3 — Step 12: Gate G1 Assessment



\*\*Decision: REVISE\*\*



\### 1. Positive images for mask calibration



All 19 classes have at least 50 positive calibration images.



\- Spalling: 588

\- Rust: 580

\- Weathering: 468

\- Rockpocket: 50 (smallest class count)



\*\*Assessment:\*\* Sufficient positive examples exist to begin calibration feasibility studies, although adequacy for a specific statistical risk guarantee has not yet been established.



\### 2. Baseline IoU and reproducibility



The mean of the best per-class calibration IoUs across the 25 frozen thresholds is 0.5105. This is an optimistic descriptive diagnostic, not an unbiased performance estimate.



Examples:

\- Bearing: 0.8388

\- EJoint: 0.7792

\- Drainage: 0.7649

\- Cavity: 0.1462

\- Rockpocket: 0.1744

\- Crack: 0.2642



Quality varies considerably by class.



Reproducibility checks passed:

\- Calibration: 1,186/1,186 identical score-file hashes

\- Test: 1,187/1,187 identical score-file hashes

\- Hash mismatches: 0



No test performance metrics were inspected or used in this assessment.



\### 3. Class mapping



All 19 class names have documented checkpoint-channel mappings.



The channel order used in `experiments/compute\_iou.py` matches `data\_cards/segmenter.md`.



Mapping consistency check: PASS.



\### 4. Gate G1 decision and rationale



\*\*G1: REVISE\*\*



The baseline is reproducible, all classes have positive calibration examples, and the implemented mappings are consistent.



However, baseline IoU is weak for several classes, and possible overlap between the checkpoint's training images and the frozen evaluation splits remains unresolved. Model preprocessing compatibility also requires validation.



\### 5. Proposed revisions before Week 4



1\. Resolve and document checkpoint training-data overlap with the professor. If necessary, use a checkpoint trained exclusively on the frozen training split.

2\. Verify inference preprocessing and compatibility against the original model implementation.

3\. Review whether weakly performing classes remain suitable for the planned calibration experiments.

4\. Preserve the frozen threshold grid and avoid using test IoU for model or threshold selection.



Proceed to Week 4's certification experiments only after the evaluation protocol is confirmed.

