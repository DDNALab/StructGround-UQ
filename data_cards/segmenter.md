

\# StructGround-UQ — Segmenter Model Card



\## 1. Selected Checkpoint



Model: SegFormer MiT-B1, dacl-challenge baseline



Architecture: SegFormerForSemanticSegmentation

with a SegModel wrapper.



Checkpoint file: best\_model\_cpu.pth



Source:

https://huggingface.co/spaces/phiyodr/dacl-challenge



Checkpoint location:

https://huggingface.co/spaces/phiyodr/dacl-challenge/tree/main/runs/2023-08-31\_rich-paper-12



Source code:

https://huggingface.co/spaces/phiyodr/dacl-challenge/blob/main/app.py



Checkpoint directory revision: 8c36ac7

Model version: 2023-08-31\_rich-paper-12



Training dataset: dacl10k

Input resolution: 512 x 512

Output channels: 19

Task: Multi-label semantic segmentation



The published application uses sigmoid probabilities

for individual output channels.



Status: Candidate checkpoint for Gate G1 review.

Not yet approved for final experiments.



\## 2. License



The Hugging Face Space does not explicitly state

a license for its released model weights.



License status: Unverified.



The dacl10k dataset license and the model-weight

license must be treated separately.



The model-weight usage rights require clarification

before the checkpoint is adopted for experiments.



\## 3. Checkpoint Output Classes



The following output channel ordering comes from

the published app.py file.



| Channel ID | Checkpoint Class |

|---|---|

| 0 | Crack |

| 1 | ACrack |

| 2 | Wetspot |

| 3 | Efflorescence |

| 4 | Rust |

| 5 | Rockpocket |

| 6 | Hollowareas |

| 7 | Cavity |

| 8 | Spalling |

| 9 | Graffiti |

| 10 | Weathering |

| 11 | Restformwork |

| 12 | ExposedRebars |

| 13 | Bearing |

| 14 | EJoint |

| 15 | Drainage |

| 16 | PEquipment |

| 17 | JTape |

| 18 | WConccor |



\## 4. Mapping to StructGround-UQ Class IDs



Our project uses alphabetical class IDs from 0 to 18.



The checkpoint uses a different output channel order.



| Project ID | Class Name | Checkpoint Channel |

|---|---|---|

| 0 | ACrack | 1 |

| 1 | Bearing | 13 |

| 2 | Cavity | 7 |

| 3 | Crack | 0 |

| 4 | Drainage | 15 |

| 5 | EJoint | 14 |

| 6 | Efflorescence | 3 |

| 7 | ExposedRebars | 12 |

| 8 | Graffiti | 9 |

| 9 | Hollowareas | 6 |

| 10 | JTape | 17 |

| 11 | PEquipment | 16 |

| 12 | Restformwork | 11 |

| 13 | Rockpocket | 5 |

| 14 | Rust | 4 |

| 15 | Spalling | 8 |

| 16 | WConccor | 18 |

| 17 | Weathering | 10 |

| 18 | Wetspot | 2 |



Exact class-name matches: 19

Missing checkpoint classes: 0

Unmatched dataset classes: 0



Mapping method: Exact string matching.



No class renaming or merging is required.



The checkpoint channel IDs must not be confused

with the project-defined class IDs.



\## 5. Gate G1 Review Items



1\. The model-weight license has not been confirmed.



2\. The checkpoint was trained on dacl10k.

&#x20;  Our frozen research splits were created from

&#x20;  the original labeled train and validation pools.



&#x20;  Therefore, calibration/test images may overlap

&#x20;  with the checkpoint's original training data.



&#x20;  This is a potential data-leakage issue and must

&#x20;  be resolved before final certification experiments.



3\. The checkpoint is stored as a PyTorch .pth

&#x20;  file containing serialized Python objects.

&#x20;  Loading such files may execute unsafe code.

&#x20;  The file should not be loaded until its

&#x20;  provenance and loading procedure are reviewed.



4\. The output class order is documented from

&#x20;  the published application source.

&#x20;  The checkpoint itself has not yet been loaded

&#x20;  or independently inspected.



Gate G1 status: PENDING.



\## 6. Training and Evaluation Restrictions



No training, fine-tuning, or threshold tuning

was performed during this step.



Do not train or tune on calibration or test data.



If training is required later, use only

splits/train.txt.



Keep the frozen test split reserved for final

evaluation and never use it for development decisions.



\## 7. Step 3 Conclusion



A publicly available dacl10k SegFormer MiT-B1

checkpoint has been identified.



All 19 class names have exact matches.



The different output channel ordering is

documented with a fixed mapping.



License and training-data overlap questions

remain pending for Gate G1 review.



