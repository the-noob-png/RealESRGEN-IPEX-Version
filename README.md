### Please check out the Orignal\Cuda version [Here](https://github.com/xinntao/Real-ESRGAN)
This version is explicitly for intel's GPUs/iGPUs/XPUs with the help of Intel’s PyTorch Extension for Pytorch or IPEX in short and intel's oneAPI.
---
## Setup the system
### Install all necessary packages
#### Please make sure that you are using kernal >=6.1,tested on 6.2
- Add Intel Graphics drivers Repository
  ```bash
  wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list && \
  sudo apt-get update
     ```
- Install packages
    ```bash
    sudo apt-get install -y \
      intel-opencl-icd intel-level-zero-gpu level-zero \
      intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
      libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
      libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
      mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo mesa-utils
    ```
 ### Verify the Installations
 - Verify the device is working with the i915 driver
   ```bash
   hwinfo --display
   ```
- Verify Media drivers installation
  ```bash
  export DISPLAY=:0.0; vainfo
  ```
### Install oneAPI Base Toolkit
- Install oneAPI
    ```bash
    # Add repo
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \ | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
    # signed entry to APT sources
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
    # Update the repo
    sudo apt-get update
    # install the intel-basekit
    sudo apt-get install intel-basekit
    # it may take upto 15GB of disk space
    ```
- Add the following lines to the end of the bash run commands file (.bashrc) for Intel’s extension to use the oneAPI toolkit
  ```bash
    export ONEAPI_ROOT=/opt/intel/oneapi
    export DPCPPROOT=${ONEAPI_ROOT}/compiler/latest
    export MKLROOT=${ONEAPI_ROOT}/mkl/latest
    export IPEX_XPU_ONEDNN_LAYOUT=1
    source ${ONEAPI_ROOT}/setvars.sh > /dev/null
     ```
  ### Run the command
  ```bash
  echo -e "\nexport ONEAPI_ROOT=/opt/intel/oneapi\nexport DPCPPROOT=\${ONEAPI_ROOT}/compiler/latest\nexport MKLROOT=\${ONEAPI_ROOT}/mkl/latest\nexport IPEX_XPU_ONEDNN_LAYOUT=1\nsource \${ONEAPI_ROOT}/setvars.sh > /dev/null" >> ~/.bashrc \
  #Source the `.bashrc`
  source ~/.bashrc
  ```
- verify GPU visibility with `sycl-ls`
    ```bash
    sycl-ls
    ```
## 🔧 Dependencies and Installation for the Script

- Python = 3.10 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- PyTorch = 2.0.1 ([xpu](https://developer.intel.com/ipex-whl-stable-xpu))

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/the-noob-png/RealESRGEN-IPEX-Version.git
    cd RealESRGEN-IPEX-Version
    ```

1. Install dependent packages

    ```bash
    # Install Torch for Intel xpus
    python3 -m pip install torch==2.0.1a0 torchvision==0.15.2a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu 
    python3 -m pip install -r requirements.txt
    python setup.py develop
    ```

---

Models:

1. RealESRGAN_x4plus  
2. RealESRNet_x4plus
3. RealESRGAN_x4plus_anime_6B
4. RealESRGAN_x2plus
5. realesr-animevideov3 (animation video) (default)
6. realesr-general-x4v3

#### Usage of python script

1. You can use X4 model for **arbitrary output size** with the argument `outscale`. The program will further perform cheap resize operation after the Real-ESRGAN output.

```console
Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

  -h                   show this help
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  -s, --outscale       The final upsampling scale of the image. Default: 4
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --face_enhance       Whether to use GFPGAN to enhance face. Default: False
  --fp32               Use fp32 precision during inference. Default: fp16 (half precision).
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

#### Inference general images

Results are in the `results` folder
