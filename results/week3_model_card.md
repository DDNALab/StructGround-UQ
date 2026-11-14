\# StructGround-UQ — Week 3 Frozen Model Card



\## Selected Checkpoint



Model: SegFormer MiT-B1 (dacl10k)



Checkpoint: best\_model\_cpu.pth



Version: 2023-08-31\_rich-paper-12



Source:

https://huggingface.co/spaces/phiyodr/dacl-challenge



Repository revision:

8c36ac7b85db6263474e17c7976f91031a069330



\## Frozen Checkpoint



Local path:

../StructGround-UQ-data/models/best\_model\_cpu.pth



SHA-256:

166b778bdbc57ef2bfd5f71bc6ae724bf598d1a4c6a1b69afa4fbd936f2d965b



The downloaded weights were checked against the

published checkpoint hash.



The checkpoint is frozen and must not be

modified or replaced during experiments.



\## License and Gate G1



Model-weight license: Not yet confirmed.



Potential overlap between the checkpoint's

training images and the frozen calibration

or test sets remains unresolved.



Gate G1 status: Pending.



\## Restrictions



\- Keep checkpoint weights outside Git.

\- Do not modify or fine-tune the weights.

\- Do not use calibration or test data for training.

\- Verify the SHA-256 hash before future experiments.

\- Do not load the serialized checkpoint until

&#x20; the loading procedure has been reviewed.




Week 3 — Step 13: Reproducibility and Model Usage Record

Checkpoint and provenance

\- Model: DACL10K challenge SegFormer MiT-B1 multi-label segmentation baseline, 19 output channels.

\- Publisher: https://huggingface.co/spaces/phiyodr/dacl-challenge

\- Source application: https://huggingface.co/spaces/phiyodr/dacl-challenge/blob/main/app.py

\- Published checkpoint: runs/2023-08-31\_rich-paper-12/best\_model\_cpu.pth (Space repository revision 8c36ac7b85db6263474e17c7976f91031a069330).

\- Local checkpoint: ../StructGround-UQ-data/models/best\_model\_cpu.pth.

\- Checkpoint SHA-256: 166b778bdbc57ef2bfd5f71bc6ae724bf598d1a4c6a1b69afa4fbd936f2d965b.

\- Weight license: not explicitly confirmed. Dataset licensing and model-weight licensing are separate issues; do not describe these weights as openly licensed without evidence.

\- PI approved proceeding with the exploratory baseline. This does not establish absence of checkpoint training-data overlap with the frozen project splits.

Frozen experimental configuration

\- Scoring config: configs/scoring.yaml, seed 2026, num\_classes: 19, output float32.

\- Threshold grid: configs/thresholds.yaml, exactly 25 frozen values 0.02, 0.06, ..., 0.98 (step 0.04). Pixels predicted positive when sigmoid score is at least the threshold.

\- Frozen lists: splits/calib.txt (1,186 images); splits/test.txt (1,187 images). Do not change these lists.

\- Model input: RGB, resize to 512 × 512, tensor conversion and ImageNet normalization, according to src/segmentation/score.py.

\- Model forward: SegFormer wrapper with nearest-neighbor ×4 upsample. Scores are obtained with sigmoid and returned at original image H × W using bilinear interpolation of logits (src/segmentation/score.py). Verify equivalence to the publisher's intended preprocessing before interpreting scientific results; it has not been independently established.

\- Outputs: one .npy array per image under results/scores/calib/ and results/scores/test/, 19 channels and original image spatial dimensions; per-split SHA-256 manifest named manifest\_sha256.txt.

\- Checkpoint loading uses PyTorch weights\_only=True with explicitly allowlisted serialized classes and an in-memory legacy Transformers config compatibility patch. The source checkpoint and weights are not edited.

Class-to-checkpoint-channel mapping

This is checkpoint channel order, not project alphabetical class IDs:

Channel	Class	Channel	Class

0	Crack	10	Weathering

1	ACrack	11	Restformwork

2	Wetspot	12	ExposedRebars

3	Efflorescence	13	Bearing

4	Rust	14	EJoint

5	Rockpocket	15	Drainage

6	Hollowareas	16	PEquipment

7	Cavity	17	JTape

8	Spalling	18	WConccor

9	Graffiti		





See data\_cards/segmenter.md for the full project-ID crosswalk. experiments/compute\_iou.py follows the above channel order.

Hardware and software

\- Windows build 26200; AMD64 Family 25 Model 24 CPU.

\- NVIDIA RTX 2000 Ada Generation GPU, 16 GB; CUDA available.

\- Conda environment: structground-w3 (Python 3.11).

\- PyTorch 2.9.1+cu128; Transformers 4.57.1; torchvision 0.24.1+cu128; NumPy 1.26.4; SciPy 1.15.3; scikit-image 0.25.2.

\- See requirements.txt and results/week3\_notes.md; record exact versions of other dependencies from the installed environment when recreating it.

Inputs required to reproduce

Clone the repository; retain its frozen splits and configuration files. Obtain the original image dataset separately; place exactly the calibration images in ../StructGround-UQ-data/calib\_images/ and exactly the test images in ../StructGround-UQ-data/test\_images/, retaining .jpg basenames. Place samples.json in ../StructGround-UQ-data/ for polygon-based IoU; this JSON is not required for scoring alone. Obtain the checkpoint from the pinned source above, place it at the local checkpoint path, and verify its SHA-256. Dataset images and score caches are not committed to Git.

Execution commands (from project root, with structground-w3 active)

python -m py\_compile experiments\\run\_scoring.py experiments\\run\_test\_scoring.py experiments\\verify\_scoring\_repro.py

python experiments\\run\_scoring.py

python experiments\\run\_scoring.py --run --g1-approved

python experiments\\run\_test\_scoring.py

python experiments\\verify\_scoring\_repro.py --split calib

python experiments\\verify\_scoring\_repro.py --split test

Important: run\_scoring.py and run\_test\_scoring.py reuse preexisting .npy files; running them a second time does not establish fresh-inference reproducibility. Use verify\_scoring\_repro.py, which computes new scores and compares their SHA-256 hashes with the original manifests without replacing the caches. Test scores may be cached and hash-verified; do not inspect test performance for research decisions.

Reproducibility result and limitations

\- Calibration: 1,186/1,186 score hashes matched on fresh inference; 0 mismatches.

\- Test: 1,187/1,187 score hashes matched on fresh inference; 0 mismatches.

\- Reproducibility reruns: calibration 812.62 s; test 1,012.72 s; combined 1,825.34 s. These include inference and hash checks; original run times were not recorded.

\- Gate G1: REVISE. Open items: potential overlap with checkpoint training images, unverified model-weight license, validation of publisher preprocessing equivalence, and class-specific weak IoU. PI approval to run exploratory experiments is recorded but does not eliminate these limitations.

