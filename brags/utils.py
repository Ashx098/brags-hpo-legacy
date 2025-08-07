# brags/utils.py
from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

# brags/utils.py
def get_gpu_utilization():
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        util = nvmlDeviceGetUtilizationRates(handle)
        return util.gpu  # percentage
    except Exception:
        return 0  # fallback
# This function initializes the NVML library, retrieves the GPU handle,
# and returns the current GPU utilization as a percentage.
# Utility function to get GPU utilization
# Returns 0 if pynvml fails (e.g., no GPU available)
# Returns a percentage (0-100) of GPU utilization
# Example usage:
# gpu_util = get_gpu_utilization()
# print(f"Current GPU Utilization: {gpu_util}%")