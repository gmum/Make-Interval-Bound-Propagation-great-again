## `method`

The directory is organized as follows:

- **get_output_bounds.py** – Experiments for robustness verification on pretrained models.  
- **train.py** – PyTorch Lightning–based training routines for standard and robustness-aware models.  
- **utils.py** – Utility functions used across the method modules, such as data processing, logging, and helper routines.  

These modules support both training and evaluation workflows, enabling computation of certified bounds and facilitating robustness experiments.