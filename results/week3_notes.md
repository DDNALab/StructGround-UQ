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




Week 3 — Step 13: Consolidated Execution Log and Handoff
What was run
1. Selected and SHA-256-verified frozen 19-channel SegFormer MiT-B1 checkpoint; documented checkpoint channel IDs in data_cards/segmenter.md and results/week3_model_card.md.
2. Froze 25 thresholds (configs/thresholds.yaml, 0.02–0.98, step 0.04) and seed 2026 (configs/scoring.yaml).
3. Verified the frozen input sets: 1,186 calibration and 1,187 test JPG images. No frozen split was modified.
4. Ran experiments/run_scoring.py --run --g1-approved for calibration; generated 1,186 score .npy files and manifest in results/scores/calib/. All 1,186 manifest hashes were independently verified with experiments/verify_scores.py.
5. Ran experiments/run_test_scoring.py for test; generated 1,187 .npy files and manifest in results/scores/test/. Test manifest filenames exactly matched the frozen test list; no test performance was inspected for decision-making.
6. Ran experiments/compute_iou.py --split calib, producing results/iou_calib.csv with columns class,threshold,IoU and exactly 19 × 25 = 475 rows. The IoU is pooled pixel intersection divided by pooled pixel union over included images at each fixed threshold.
7. Ran experiments/compute_iou.py --split test, producing results/iou_test.csv with exactly 475 rows. Test IoU values were not displayed, reviewed or used for decisions; that file is excluded from Git.
8. Ran experiments/verify_scoring_repro.py --split calib and --split test to perform fresh inference and match hashes without overwriting originals.
Annotation and evaluation conventions
- Ground truth comes from ../StructGround-UQ-data/samples.json, rasterized using src/segmentation/union_mask.py.
- Calibration: 1,177 images with polygon annotations included; 9 no-polygon images excluded as annotation-ambiguous.
- Test: 1,181 images with polygon annotations included; 6 no-polygon images excluded. Test IoU values remain sealed.
- These exclusions are documented conventions, not evidence that omitted images are damage-free. Confirm the convention with the PI before final interpretation.
- Scores are sigmoid predictions per channel; experiments/compute_iou.py uses checkpoint channel order rather than project alphabetical IDs.
Calibration-only IoU diagnostic
- Each class has at least 50 positive calibration images (19/19 classes at least 30).
- Mean of maximum per-class calibration IoU over the 25 frozen thresholds = 0.5105. This is an optimistic descriptive maximum, not an unbiased estimate and not a permitted basis for test tuning.
- Best calibration IoUs for selected classes: Bearing 0.8388; EJoint 0.7792; Drainage 0.7649; PEquipment 0.7257; Cavity 0.1462; Rockpocket 0.1744; Crack 0.2642.
- The complete 19-class × 25-threshold calibration table is results/iou_calib.csv. Test IoU remains excluded from decision-making.
Cost and reproducibility
- Hardware: Windows build 26200, AMD64 Family 25 Model 24, NVIDIA RTX 2000 Ada Generation 16 GB GPU; PyTorch 2.9.1+cu128; Transformers 4.57.1; seed 2026.
- Disk size of original score caches: calibration 368.84 GB; test 385.39 GB; combined 754.23 GB (at measurement).
- Original score-run times not measured. Fresh-inference reproducibility reruns (include loading/inference and SHA-256 checks): calibration 812.62 s (13 min 32.62 s); test 1,012.72 s (16 min 52.72 s); total 1,825.34 s (30 min 25.34 s).
- New-to-original .npy SHA-256 matches: calibration 1,186/1,186; test 1,187/1,187; 0 mismatches. Summaries: results/week3_repro_calib.json, results/week3_repro_test.json.
- For actual rerun instructions, local file layout, model download revision and SHA-256, frozen threshold values and channel mapping, see the Step 13 reproducibility section in results/week3_model_card.md.
Problems encountered
- The 2023 checkpoint contains serialized Python objects and needed restricted allowlisting, including the custom SegModel name when the module was imported by the test runner.
- Transformers 4.57.1 required legacy configuration attributes to be populated in memory before inference. The model weights and saved checkpoint were not changed.
- The existing score runner silently reuses cached .npy outputs; fresh-inference reproducibility therefore required a separate script.
- Original score-run times were not recorded; only reproducibility-run times are available.
- Potential leakage: checkpoint trained on DACL10K, whereas the frozen project calibration/test splits include original DACL10K training images. Exact overlap remains unresolved.
- The model-weight license is unverified, and equivalence of image preprocessing/output handling to publisher inference needs independent confirmation.
- Low calibration IoU for some classes; annotation-empty images omitted rather than assumed negative.
Gate G1 decision
REVISE. Sufficient positive calibration data exists, class names/channel indices are consistently documented, and file-hash reproducibility passed. However, checkpoint training-data overlap, license, preprocessing equivalence, and uneven class IoU require resolution before Week 4 certification results are represented as valid.

Changes proposed before Week 4: (1) Have the PI document training image provenance or switch/retrain to a strictly train-only checkpoint; (2) verify published preprocessing and probability semantics against score.py; (3) confirm annotation-empty image policy and weak-class inclusion; (4) preserve frozen splits/grid and do not use sealed test IoU for tuning.
Commands for new researchers
See the Step 13 reproducibility and execution instructions in results/week3_model_card.md; all scripts run from the repository root in Conda environment structground-w3. The checkpoint and JPG files are separate downloads; the two Markdown documents alone are not substitutes for the code and data assets.

