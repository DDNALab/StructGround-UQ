\# StructGround-UQ — Week 2 Progress Notes



\*\*Project:\*\* StructGround-UQ  

\*\*Week:\*\* Week 2  

\*\*Dataset:\*\* dacl10k (Hugging Face version)  

\*\*Status:\*\* Implementation completed; final combined test verification pending



\## 1. Dataset Summary



The dacl10k dataset contains 8,922 image records distributed across three original dataset splits.



| Original split | Images |

|---|---:|

| Train | 6,935 |

| Validation | 975 |

| Testdev | 1,012 |

| \*\*Total\*\* | \*\*8,922\*\* |



The original testdev split does not contain segmentation ground-truth annotations and was excluded from the research splitting procedure.



\### Frozen Research Splits



A fixed random seed of 2026 was used to divide the 7,910 labeled images into three research sets.



| Research split | Images |

|---|---:|

| Train | 5,537 |

| Calibration | 1,186 |

| Test | 1,187 |

| \*\*Total\*\* | \*\*7,910\*\* |



The splits were saved in `splits/train.txt`, `splits/calib.txt`, and `splits/test.txt`.



SHA-256 hashes were saved in `splits/sha256.txt`, and all three hashes were verified successfully.



No reliable inspection-level or source-level grouping identifier was established during Week 1. Therefore, image-level random splitting was used. This does not guarantee independence between related images from the same inspection.



The research test split is reserved for final evaluation and must not be used for model development or calibration decisions.



\## 2. Dataset Class Distribution



The following counts represent images containing each class across the original labeled dataset (train and validation combined). An image may contain more than one damage class.



| Class | Images |

|---|---:|

| ACrack | 374 |

| Bearing | 835 |

| Cavity | 1,355 |

| Crack | 1,981 |

| Drainage | 1,181 |

| EJoint | 446 |

| Efflorescence | 1,729 |

| ExposedRebars | 877 |

| Graffiti | 941 |

| Hollowareas | 1,252 |

| JTape | 895 |

| PEquipment | 1,494 |

| Restformwork | 846 |

| Rockpocket | 363 |

| Rust | 3,915 |

| Spalling | 3,739 |

| WConccor | 303 |

| Weathering | 3,096 |

| Wetspot | 1,109 |



\*\*Total unique classes:\*\* 19



Class counts overlap because a single image can contain several annotated classes. These are original-dataset counts, not separate class counts for each newly frozen research split.



\## 3. Per-Class Union Masks



Implemented `src/segmentation/union\_mask.py` to combine all filled polygons of the same class into one binary mask per image.



The implementation was validated using synthetic examples and a real dacl10k annotation, `dacl10k\_v2\_train\_1394`.



The real example contained five annotated damage classes: Cavity, Efflorescence, Hollowareas, Spalling, and WConccor.



\## 4. Connected Components and Bounding Boxes



Implemented `src/segmentation/bounding\_boxes.py`.



The code detects connected components and calculates one tight enclosing rectangle per image–class mask.



\*\*Generated records:\*\*



\- Training: 23,346 image–class box records

\- Validation: 3,385 image–class box records

\- Total: 26,731 unique image–class records

\- Empty generated class masks: 0



The bounding boxes were saved to `StructGround-UQ-data/dacl10k\_boxes.csv` before any experimental mask hiding.



The current implementation uses a single enclosing box around all components. This differs from the union of component-level tight boxes described in the manuscript and must be reconciled before final statistical experiments.



\## 5. Occupancy Ratio Analysis



Occupancy was calculated as:



\*\*Occupancy = damaged mask pixels / enclosing bounding-box area\*\*



The values were saved in `StructGround-UQ-data/dacl10k\_occupancy.csv`.



All 26,731 records passed the occupancy-range validation.



\*\*Observed range:\*\* 0.00050476 to 1.0



\### Median Occupancy by Class



| Class | Median occupancy |

|---|---:|

| ACrack | 0.5948 |

| Bearing | 0.7031 |

| Cavity | 0.0450 |

| Crack | 0.1259 |

| Drainage | 0.6503 |

| EJoint | 0.6184 |

| Efflorescence | 0.3184 |

| ExposedRebars | 0.2818 |

| Graffiti | 0.5020 |

| Hollowareas | 0.7095 |

| JTape | 0.4266 |

| PEquipment | 0.5235 |

| Restformwork | 0.3762 |

| Rockpocket | 0.4033 |

| Rust | 0.2262 |

| Spalling | 0.2901 |

| WConccor | 0.5251 |

| Weathering | 0.5964 |

| Wetspot | 0.4961 |



Cavity and Crack have relatively low median occupancy, while Hollowareas and Bearing have relatively high occupancy.



These measurements may help analyze how damage geometry affects weak bounding-box information during later experiments.



\## 6. Weak-Box Loss Interval Function



Implemented `src/risk/interval.py` using the three cases specified in the research instructions.



\- Full bounding-box coverage: interval \[0, 0]

\- No bounding-box coverage: interval \[1, 1]

\- Partial bounding-box coverage: interval \[0, 1]



All four synthetic unit tests passed.



Fifteen additional checks using five real dacl10k damage masks and constructed prediction regions passed. These are integration tests, not evaluations of a trained segmentation model.



\## 7. Calibration Mask/Box Random Partition



Implemented `src/risk/partition.py`.



For a calibration pool and mask budget `n`, the function randomly assigns `n` images to the revealed-mask set and the remaining images to the box-only set.



The partition uses image names only.



Six synthetic unit tests passed.



Integration testing with the actual 1,186-image calibration pool succeeded for three different replication seeds:



| Seed | Mask images | Box-only images |

|---|---:|---:|

| 2026 | 100 | 1,086 |

| 2027 | 100 | 1,086 |

| 2028 | 100 | 1,086 |



All tested partitions were disjoint and covered the full calibration pool.



Future data-loading code must enforce the restriction against reading hidden masks.



\## 8. Testing Summary



Previously executed tests:



| Test file | Result |

|---|---|

| `tests/test\_union\_mask.py` | 3 passed |

| `tests/test\_bounding\_boxes.py` | 3 passed |

| `tests/test\_interval.py` | 4 passed |

| `tests/test\_partition.py` | 6 passed |

| `tests/check\_real\_mask.py` | Passed |

| `tests/check\_real\_boxes.py` | Passed |

| `tests/check\_real\_interval.py` | 15 integration checks passed |

| `tests/check\_real\_partition.py` | 3 replication checks passed |



\*\*Total synthetic unit tests previously passed:\*\* 16



The complete unit-test suite must be rerun together as the final Week 2 verification.



\## 9. Blockers and Limitations



1\. No reliable inspection/source grouping identifier was identified. Splits were created at image level, leaving possible dependence between related images.

2\. Some original annotations contained normalized polygon coordinates outside the expected \[0, 1] range. The existing rasterization implementation requires further validation of its handling of these coordinates.

3\. The current enclosing-box representation differs from the manuscript's union-of-component-boxes representation.

4\. The full image dataset remains in Google Drive; local preprocessing uses annotation data and saved derived records.

5\. The permission/license information should be clarified before redistributing original dataset material.

6\. Actual segmentation-model predictions are not yet available. The interval function has been tested using constructed prediction regions.

7\. Future experiments must enforce hidden-mask access restrictions, not merely produce disjoint image lists.



These are documented limitations or future-work requirements. They did not prevent completion of the Week 2 preprocessing and function-level tests.



\## 10. Week 2 Completion Checklist



\- \[x] Create project structure and data card

\- \[x] Implement per-class binary union masks

\- \[x] Implement connected components and bounding boxes

\- \[x] Generate and save bounding boxes

\- \[x] Compute and save occupancy ratios

\- \[x] Implement weak-box loss intervals

\- \[x] Freeze train/calibration/test splits

\- \[x] Save and verify SHA-256 hashes

\- \[x] Implement calibration mask/box random partition

\- \[x] Run individual synthetic and real-data checks

\- \[ ] Run final combined unit-test suite

\- \[ ] Commit verified Week 2 notes



\*\*Week 2 conclusion:\*\* The required preprocessing functions, interval calculations, random partitioning, and frozen data splits have been implemented. Final combined testing and documentation commit remain before formal Week 2 completion.

