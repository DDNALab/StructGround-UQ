\# dacl10k Dataset Card



\*\*Project:\*\* StructGround-UQ  

\*\*Dataset:\*\* dacl10k v2  

\*\*Data source:\*\* Hugging Face — Voxel51/dacl10k  

\*\*Purpose:\*\* Structural damage segmentation and risk calibration  

\*\*Documentation date:\*\* November 2026



\---



\## 1. Dataset Overview



dacl10k is a bridge-damage image dataset containing structural damage annotations.



For StructGround-UQ, it will be used to evaluate whether bounding-box annotations can reduce the number of exact segmentation masks required for statistically reliable segmentation risk certification.



The dataset was downloaded from Hugging Face after the original AWS download link returned a 403 Forbidden error.



\*\*Dataset links:\*\*



\- Official repository: https://github.com/phiyodr/dacl10k-toolkit

\- Hugging Face: https://huggingface.co/datasets/Voxel51/dacl10k



\## 2. Dataset Size and Splits



| Split | Images | Annotated Polygons |

|---|---:|---:|

| Training | 6,935 | 54,158 |

| Validation | 975 | 8,167 |

| Testdev | 1,012 | 0 |

| \*\*Total\*\* | \*\*8,922\*\* | \*\*62,325\*\* |



All 8,922 image files were verified to exist.



The testdev split contains images but does not provide segmentation ground-truth annotations in this release.



\## 3. Annotation Classes



The following 19 class names were extracted directly from the Hugging Face annotation records.



Image counts represent the number of images containing at least one polygon of that class, across the labeled training and validation splits.



| Class | Images | Polygons |

|---|---:|---:|

| Rust | 3,915 | 14,720 |

| Spalling | 3,739 | 9,884 |

| Weathering | 3,096 | 4,634 |

| Crack | 1,981 | 3,626 |

| Efflorescence | 1,729 | 3,965 |

| PEquipment | 1,494 | 1,888 |

| Cavity | 1,355 | 9,271 |

| Hollowareas | 1,252 | 1,557 |

| Drainage | 1,181 | 1,652 |

| Wetspot | 1,109 | 1,661 |

| Graffiti | 941 | 2,200 |

| JTape | 895 | 1,078 |

| ExposedRebars | 877 | 2,000 |

| Restformwork | 846 | 1,022 |

| Bearing | 835 | 1,208 |

| EJoint | 446 | 506 |

| ACrack | 374 | 424 |

| Rockpocket | 363 | 617 |

| WConccor | 303 | 412 |



\*\*Total unique annotation classes: 19.\*\*



Some classes represent damage, while others represent structural objects or associated features.



The class names are preserved exactly as observed in the annotation files.



\## 4. Annotation and Mask Format



\*\*Annotation type:\*\* Multi-label semantic segmentation.



The Hugging Face version stores annotations in a consolidated `samples.json` file using FiftyOne-style polygon records.



Important fields include:



\- `filepath`: Image file location

\- `tags`: Dataset split

\- `metadata`: Image dimensions and file details

\- `semseg\_ground\_truth\_filled`: Polygon segmentation annotations

\- `label`: Annotation class

\- `points`: Polygon coordinates

\- `closed` and `filled`: Polygon properties



Polygon coordinates are normalized relative to the image dimensions.



Multiple polygons can share the same class label within one image. For semantic segmentation, polygons belonging to the same class will be combined into a class-level mask.



The dataset does not provide verified, persistent defect-instance identities for every polygon.



\*\*Week 1 visualization:\*\* Successfully inspected `dacl10k\_v2\_train\_1394.jpg`, containing seven polygons across five annotation classes.



\## 5. Annotation Integrity Findings



An initial validation of the downloaded records found:



\- Missing image files: 0

\- Training images with empty polygon lists: 62

\- Validation images with empty polygon lists: 9

\- Testdev images without ground-truth segmentation annotations: 1,012

\- Polygons containing coordinates outside the normalized \[0, 1] range: 1,059

\- Images affected by these coordinate deviations: 951



The out-of-range polygons include 938 training polygons and 121 validation polygons.



These findings do not establish that the original annotations are incorrect. The source of the coordinate deviations has not been confirmed.



No annotation files were modified during Week 1.



\## 6. Source and Group Identifiers



The following top-level metadata fields were inspected:



`\_dataset\_id`, `\_id`, `\_media\_type`, `\_rand`, `created\_at`, `created\_date`, `filepath`, `metadata`, `multi\_ground\_truth`, `name`, `od\_ground\_truth`, `semseg\_ground\_truth\_filled`, and `tags`.



\*\*Finding:\*\* No explicit bridge ID, inspection ID, source ID, or group ID was identified among the top-level fields.



This does not prove that images are independent or that near-duplicates are absent.



Before experimental splits are frozen, related images and available provenance information should be investigated to reduce potential data leakage.



\## 7. License and Usage Restrictions



The official dacl10k repository lists \*\*Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)\*\*.



The Hugging Face dataset listing reports \*\*CC BY 4.0\*\*.



This discrepancy has not been resolved with the dataset owners.



Until clarified, the project will conservatively follow the original non-commercial restriction and avoid publicly redistributing dataset images, masks, or derived crops without confirming permission.



Research scripts, configurations, and aggregate results can be maintained separately from the original dataset files.



\## 8. Official AWS vs. Hugging Face Comparison



| Property | Finding |

|---|---|

| Image split counts | Match published official counts |

| Annotation class names | All 19 documented classes observed |

| Annotation storage | Hugging Face uses consolidated FiftyOne JSON |

| Coordinate representation | Hugging Face uses normalized coordinates |

| Exact image equivalence | Not verified |

| Exact polygon equivalence | Not verified |

| License listings | Inconsistent |



The Hugging Face version appears suitable for further preprocessing investigation, but exact equivalence with the original AWS annotation files has not been established.



\## 9. Research Usage



In the StructGround-UQ project, dacl10k will provide existing dense annotations for:



\- Constructing class-specific segmentation masks.

\- Automatically deriving weak bounding-box annotations.

\- Creating independent calibration and evaluation datasets.

\- Comparing mask-only and mixed-annotation risk-calibration methods.



No new segmentation masks will be manually annotated.



The original downloaded dataset must remain separate from the Git repository.



\## 10. Week 1 Conclusion



The complete Hugging Face dacl10k dataset was downloaded and verified. Dataset splits, all 19 annotation classes, class frequencies, and polygon structure were inspected.



The dataset includes usable polygon annotations for training and validation images, while testdev ground-truth masks are unavailable.



Source-group information, licensing discrepancies, and coordinate deviations have been documented for further review before finalizing the experimental preprocessing protocol.





\## Frozen Research Splits



\- Dataset: Hugging Face dacl10k

\- Eligible images: 7,910 labeled images

\- Original train: 6,935 images

\- Original validation: 975 images

\- Original testdev: excluded because segmentation ground truth is unavailable

\- Random seed: 2026

\- Split strategy: image-level random assignment

\- Train: 5,537 images (70%)

\- Calibration: 1,186 images (approximately 15%)

\- Test: 1,187 images (approximately 15%)



\### Source and Inspection Grouping



No reliable source-level or inspection-level grouping

identifier was established during the Week 1 metadata review.



Therefore, each unique image name was treated as one

group when creating the frozen splits.



Limitation: image-level splitting does not guarantee

independence between visually similar images or images

from the same inspection.



\### Reproducibility



Frozen split files:

\- splits/train.txt

\- splits/calib.txt

\- splits/test.txt



SHA-256 manifest:

\- splits/sha256.txt



The split files must not be regenerated or modified.

The test split is reserved for final evaluation and

must not be inspected to guide development decisions.



Bounding-box and occupancy preprocessing was completed

before these research splits were frozen.





## StructGround-UQ Class ID Mapping

The Hugging Face dacl10k polygon annotations use string
class labels. No authoritative numeric class-ID mapping
has been established from the inspected annotation records.

For reproducibility, StructGround-UQ assigns the following
project-specific IDs in alphabetical order.

| Project ID | Exact Class Name |
|---|---|
| 0 | ACrack |
| 1 | Bearing |
| 2 | Cavity |
| 3 | Crack |
| 4 | Drainage |
| 5 | EJoint |
| 6 | Efflorescence |
| 7 | ExposedRebars |
| 8 | Graffiti |
| 9 | Hollowareas |
| 10 | JTape |
| 11 | PEquipment |
| 12 | Restformwork |
| 13 | Rockpocket |
| 14 | Rust |
| 15 | Spalling |
| 16 | WConccor |
| 17 | Weathering |
| 18 | Wetspot |

These IDs are project-defined and must not be interpreted
as official dataset category IDs.

The mapping must remain unchanged throughout subsequent
experiments. The original string labels remain authoritative.


