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



