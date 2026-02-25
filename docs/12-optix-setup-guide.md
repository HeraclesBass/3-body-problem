# OptiX SDK Setup Guide for GH200

## Step-by-Step Installation

### 1. Download OptiX SDK (On your GPU server or local machine)

**NVIDIA Developer Portal:**
https://developer.nvidia.com/designworks/optix/downloads

**What to download:**
- **OptiX SDK 8.1.0** (or latest 8.x version)
- Platform: **Linux**
- Architecture: **ARM64** (for GH200)

**File will be named something like:**
```
NVIDIA-OptiX-SDK-8.1.0-linux-ARM64.sh
```

**Note:** The download requires login with your NVIDIA Developer account.

### 2. Transfer to GH200

```bash
# From the server (or wherever you downloaded it)
scp -i ~/.ssh/your_key \
  ~/Downloads/NVIDIA-OptiX-SDK-8.1.0-linux-ARM64.sh \
  ubuntu@<GPU_HOST_IP>:~/
```

### 3. Install OptiX SDK on GH200

```bash
# SSH to GH200
ssh -i ~/.ssh/your_key ubuntu@<GPU_HOST_IP>

# Make installer executable
chmod +x NVIDIA-OptiX-SDK-8.1.0-linux-ARM64.sh

# Run installer (accept license, use default install path)
./NVIDIA-OptiX-SDK-8.1.0-linux-ARM64.sh

# Default install location: ~/NVIDIA-OptiX-SDK-8.1.0/
```

### 4. Set Environment Variables

```bash
# Add to ~/.bashrc on GH200
cat >> ~/.bashrc << 'EOF'

# OptiX SDK
export OPTIX_ROOT=$HOME/NVIDIA-OptiX-SDK-8.1.0
export PATH=$OPTIX_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$OPTIX_ROOT/lib64:$LD_LIBRARY_PATH

# CUDA (should already be installed on Lambda)
export CUDA_ROOT=/usr/local/cuda
export PATH=$CUDA_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH

EOF

# Reload bashrc
source ~/.bashrc
```

### 5. Verify OptiX Installation

```bash
# Check OptiX headers exist
ls $OPTIX_ROOT/include/optix*.h

# Expected output:
# optix.h
# optix_device.h
# optix_function_table.h
# optix_host.h
# optix_stack_size.h
# optix_types.h
```

### 6. Install Python Dependencies

```bash
# Create/activate venv for project
cd ~/3-body-problem-v8
python3 -m venv venv
source venv/bin/activate

# Install Warp first (needed for physics)
pip install warp-lang

# Install OptiX Python bindings
pip install pyoptix

# Install other dependencies
pip install numpy pillow librosa
```

### 7. Test PyOptiX Installation

```bash
# Test import
python3 << 'EOF'
import pyoptix as optix
print(f"PyOptiX version: {optix.__version__}")
print(f"OptiX available: {optix.is_available()}")

# Try to initialize OptiX context
try:
    ctx = optix.DeviceContext()
    print("✅ OptiX context created successfully!")
    print(f"Device count: {ctx.get_device_count()}")
except Exception as e:
    print(f"❌ Error: {e}")
EOF
```

**Expected output:**
```
PyOptiX version: X.X.X
OptiX available: True
✅ OptiX context created successfully!
Device count: 1
```

### 8. Test Warp Installation

```bash
python3 << 'EOF'
import warp as wp

wp.init()
print(f"Warp version: {wp.__version__}")
print(f"CUDA available: {wp.is_cuda_available()}")

if wp.is_cuda_available():
    device = wp.get_cuda_device()
    print(f"CUDA device: {device}")

    # Get GPU name
    try:
        props = wp.get_cuda_device_properties(0)
        print(f"GPU: {props['name']}")
        print(f"Memory: {props['total_global_mem'] / (1024**3):.1f} GB")
    except:
        print("Could not get device properties")
EOF
```

**Expected output:**
```
Warp version: 1.X.X
CUDA available: True
CUDA device: cuda:0
GPU: NVIDIA GH200 Grace Hopper Superchip
Memory: 96.0 GB
```

## Troubleshooting

### Issue: "OptiX not available"

**Check CUDA installation:**
```bash
nvcc --version
nvidia-smi
```

**Check OptiX headers:**
```bash
ls $OPTIX_ROOT/include/optix.h
```

**Check driver version (needs 470+):**
```bash
cat /proc/driver/nvidia/version
```

### Issue: "Module 'pyoptix' not found"

**Try alternative installation:**
```bash
# Alternative 1: python-optix (different package)
pip install python-optix

# Alternative 2: Install from source
git clone https://github.com/NVIDIA/otk-pyoptix.git
cd otk-pyoptix
pip install .
```

### Issue: CUDA not found

**Lambda instances should have CUDA pre-installed. If not:**
```bash
# Find CUDA
find /usr/local -name "cuda*" -type d 2>/dev/null

# Or check Lambda's environment
env | grep CUDA
```

## Next Steps After Installation

Once everything is installed and verified:

1. **Run OptiX samples** (if included)
   ```bash
   cd $OPTIX_ROOT/SDK
   ls -la  # Look for example programs
   ```

2. **Create minimal test** (our first OptiX program)
   - Initialize OptiX context
   - Create simple scene
   - Render to buffer
   - Save as PNG

3. **Clone optixParticleVolumes example**
   ```bash
   cd ~/
   git clone https://github.com/nvpro-samples/optix_advanced_samples.git
   cd optix_advanced_samples/src/optixParticleVolumes
   ```

4. **Begin bridge implementation**
   - Test CUDA pointer sharing between Warp and OptiX
   - Verify zero-copy works

## Installation Checklist

- [ ] OptiX SDK downloaded from NVIDIA
- [ ] SDK transferred to GH200
- [ ] SDK installed (`~/NVIDIA-OptiX-SDK-8.1.0/`)
- [ ] Environment variables set in `~/.bashrc`
- [ ] Python venv created
- [ ] Warp installed and tested
- [ ] PyOptiX installed and tested
- [ ] OptiX context creation successful
- [ ] CUDA device detection working
- [ ] Ready for minimal OptiX program!

---

**Time estimate:** 30-45 minutes (mostly download/transfer time)
