import cupy as cp
import time

# CUDA kernel that hogs GPU
kernel_code = r'''
extern "C" __global__
void high_load_kernel(float* out, int iterations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        val += sinf(i * 0.0001f) * cosf(i * 0.00005f);
    }

    out[idx] = val;
}
'''

# Compile kernel once
module = cp.RawModule(code=kernel_code)
kernel = module.get_function('high_load_kernel')

# Simulation parameters
NUM_THREADS = 1024 * 1024   # 1 million
ITERATIONS = 100000         # simulate expensive computation
BLOCK_SIZE = 256
GRID_SIZE = (NUM_THREADS + BLOCK_SIZE - 1) // BLOCK_SIZE
TARGET_FPS = 25
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Allocate output buffer
output = cp.zeros(NUM_THREADS, dtype=cp.float32)

print("⚠️  Starting GPU overload simulation at 25 FPS...")
try:
    while True:
        start_time = time.time()

        # Launch synthetic high-load kernel
        kernel((GRID_SIZE,), (BLOCK_SIZE,), (output, cp.int32(ITERATIONS)))
        cp.cuda.Device().synchronize()

        # Enforce real-time frame pacing
        elapsed = time.time() - start_time
        sleep_time = max(0, FRAME_INTERVAL - elapsed)
        time.sleep(sleep_time)

        print(f"Frame done in {elapsed:.3f} s | sleeping {sleep_time:.3f} s")

except KeyboardInterrupt:
    print("❎ Simulation stopped.")

