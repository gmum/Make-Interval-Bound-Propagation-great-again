## `crown_ibp_model_defs`

The directory is organized as follows:

- **conv4layers.py** – Defines a simple convolutional network with four layers, suitable for small-scale experiments.  
- **dm_large.py** – Implements the large variant of the DeepMind architectures used in the CROWN-IBP paper, designed for high-capacity tasks.  
- **dm_medium.py** – Implements the medium variant of the DeepMind architectures, balancing model size and performance.  
- **dm_small.py** – Implements the small variant of the DeepMind architectures, optimized for faster training and lower memory usage.  

All of these architectures were used in the [CROWN-IBP](https://arxiv.org/pdf/1906.06316) paper to evaluate provable robustness against adversarial attacks on different network sizes and complexities. They serve as reference implementations for experimenting with certified robustness using interval bound propagation methods.