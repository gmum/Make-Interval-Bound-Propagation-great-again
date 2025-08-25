## `method`

The directory is organized as follows:

- **affine_arithmetic.py** – Implements the affine arithmetic method for tighter bounds in robustness verification.  
- **alpha_crown.py** – Implements the $\alpha$-CROWN algorithm for computing certified lower bounds on adversarial robustness.  
- **composer.py** – Provides utilities to compose multiple robustness verification methods.  
- **crown.py** – Implements the standard CROWN method.  
- **ibp.py** – Implements Interval Bound Propagation (IBP) for efficient certified robustness evaluation.  
- **interval_arithmetic.py** – Implements interval arithmetic operations for bound propagation in neural networks.  
- **lower_bound.py** – Contains utilities and functions to compute lower bounds for network outputs.  
- **method_plugin_abc.py** – Defines the abstract base class for robustness verification methods, allowing easy extension with new techniques.  

These modules collectively provide implementations of various certified robustness methods, including CROWN, IBP, and their variants, as well as utilities for composing and extending verification strategies.
