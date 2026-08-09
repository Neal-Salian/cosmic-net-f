import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401  (must load BEFORE numpy/MKL DLLs on Windows, or torch's shm.dll fails to load)
