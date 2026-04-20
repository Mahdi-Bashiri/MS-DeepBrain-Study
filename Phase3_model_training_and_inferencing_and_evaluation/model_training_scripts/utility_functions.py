"""
P4 Article - Utility Functions


Developer:
"Mahdi Bashiri Bawil"
"""

import gc
import tensorflow as tf
from tensorflow.keras import backend as K

print("TensorFlow Version:", tf.__version__)

###################### GPU Configuration ######################

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("✅ GPU memory growth enabled")
        print(f"   Available GPUs: {len(physical_devices)}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠️  No GPU detected - training will be slow")
 
"""
GPU Memory Management for Sequential Experiments
To properly release memory between experiments
"""


def clear_gpu_memory():
    """
    Comprehensive GPU memory cleanup between experiments
    Call this after each experiment completes
    """
    print("\n" + "="*70)
    print("CLEANING UP GPU MEMORY")
    print("="*70)
    
    # Clear Keras session
    K.clear_session()
    print("✅ Cleared Keras session")
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    print("✅ Ran garbage collection (3 passes)")
    
    # Reset TensorFlow graphs
    tf.compat.v1.reset_default_graph()
    print("✅ Reset default graph")
    
    # Additional cleanup for TF 2.x
    try:
        # Clear any cached tensors
        tf.config.experimental.reset_memory_stats('GPU:0')
        print("✅ Reset GPU memory stats")
    except:
        pass
    
    # CRITICAL: Reset GPU memory allocator
    # This forces TensorFlow to release memory back to the system
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            # Disable and re-enable memory growth to flush allocator
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, False)
                tf.config.experimental.set_memory_growth(device, True)
            print("✅ Reset memory growth (flushed allocator)")
    except Exception as e:
        print(f"⚠️  Could not reset memory growth: {e}")
    
    print("="*70 + "\n")


def get_gpu_memory_info():
    """
    Print current GPU memory usage
    Useful for monitoring memory leaks
    """
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            for device in gpu_devices:
                details = tf.config.experimental.get_memory_info(device.name.replace('/physical_device:', ''))
                current_mb = details['current'] / 1024**2
                peak_mb = details['peak'] / 1024**2
                print(f"GPU Memory - Current: {current_mb:.1f} MB, Peak: {peak_mb:.1f} MB")
    except Exception as e:
        print(f"Could not get GPU memory info: {e}")
