# Axion: A multi-slice context UNet model, for high-resolution segmentation.

What it is: A modular, configuration-driven deep learning pipeline for binary image segmentation, built with a strong emphasis on ETL rigor, reproducibility, and stage-wise observability.

What it does: Converts volumetric scans into context-aware 2.5D representations, trains a UNet-based segmentation model with hybrid loss optimization, and produces calibrated predictions with spatial overlay visualizations.

What's unique?: Instead of treating each slice independently, the model operates on a sliding window of adjacent slices (prev, current, next), introducing local volumetric context. This reduces fragmentation and improves stability in segmentation outputs, particularly for small or discontinuous regions.

---
###  Architecture: -
<img width="1326" height="416" alt="image" src="https://github.com/user-attachments/assets/fbccd0fe-665d-4f54-a4d3-b816fc0bf039" />

```
How it works:

1. Raw 3D MRI volumes are converted into 2.5D slices using OpenCV—neighboring slices are stacked as channels to inject
   local depth context.

2. The slices are resized, normalized, and split at the subject level (not slice-wise) to prevent data leakage.

3. A UNet is trained on these multi-channel inputs to predict binary segmentation masks, using a Dice + BCE loss for
   stable optimization under class imbalance.

4. The encoder captures coarse spatial structure, while the decoder reconstructs fine boundaries via skip connections.

5. Training runs with mixed precision and LR scheduling, with Dice and IoU tracked per epoch for convergence.

6. During inference, logits are passed through sigmoid + thresholding to produce masks, and OpenCV is used again to
   generate overlay visualizations for quick inspection.
```
---


### Structure Dynamics: -

The reason I engineered this 2.5d segmentation architectue: -

1. 2.5D inputs encode local volumetric context by stacking neighboring slices as channels. This
   allows the model to see continuity across slices without the cost of full 3D processing.
   
3. The UNet encoder captures hierarchical spatial features, learning coarse structures
   such as tumor regions and their approximate boundaries.

4. The decoder refines these features using skip connections, recovering fine-grained
   details and sharp segmentation boundaries that are often lost during downsampling.

5. The hybrid loss (Dice + BCE) balances global overlap quality with pixel-wise accuracy,
   making the model robust to class imbalance and small tumor regions.

6. The result is a representation that jointly captures local context, global structure,
   and boundary precision, producing stable and interpretable segmentation masks.
---

### Results: - 
1. Dice score: -
<img width="786" height="318" alt="image" src="https://github.com/user-attachments/assets/c9ae9f3b-739f-4cd5-8aa3-500ee714f994" />
  
2. IoU score: -
<img width="786" height="318" alt="image" src="https://github.com/user-attachments/assets/911411de-71f2-44ac-be11-7133e0c67341" />

3.
 | Model        | Dice  | IoU  |
|--------------|--------|-------|
| UNet (2.5D)  | 0.842  | 0.733 |

4.
    a. The 2.5D UNet captures spatial continuity across slices, showing that incorporating local depth context improves segmentation stability compared to isolated        2D inputs.
    b. The model learns both coarse structure and fine boundaries through the encoder-decoder design, indicating that hierarchical feature extraction is sufficient        to localize tumor regions effectively.
    c. The hybrid Dice + BCE loss improves performance by balancing overlap quality and pixel-wise accuracy, leading to more consistent predictions, especially on smaller or irregular regions.

### How to Run: -

  1. Install the [dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation/code) to /data/raw
  2. Execute the cmd in terminal, create a .venv, install the requirements and then execute the main.
    ```
    python -m venv venv
    & <path_to_venv>\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    cd <path/to/mainfolder>
    python -m src.main
    ```

   
