import tensorflow as tf
import sys

def verify_gpu_installation():
    """
    Checks TensorFlow and GPU device visibility.
    """
    print("--- TensorFlow GPU Verification ---")

    # 1. Check TensorFlow Version
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Version: {sys.version}")

    # 2. Check GPU Devices
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"\nSUCCESS: Found {len(gpus)} GPU device(s).")
        for gpu in gpus:
            print(f"-> Device Name: {gpu.name}")
            print(f"-> Device Type: {gpu.device_type}")

        # Optional: Test a simple operation (Tensor addition on GPU)
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([1.0, 2.0, 3.0], shape=[3, 1])
                b = tf.constant([4.0, 5.0, 6.0], shape=[3, 1])
                c = a + b
                print("\nSimple Tensor Test (Running on GPU:0):")
                print(f"Result (a + b):\n{c.numpy()}")
        except RuntimeError as e:
            print(f"\nWARNING: Could not run tensor test on GPU: {e}")
            print("This sometimes indicates an environment variable or cuDNN issue.")

    else:
        print("\nFAILURE: TensorFlow did not find any GPU devices.")
        print("Please double-check your CUDA Toolkit, cuDNN SDK, and NVIDIA driver versions.")
        print("For Windows, ensure you are using WSL2.")

if __name__ == '__main__':
    verify_gpu_installation()
