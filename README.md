# Investigating text-guided cross-region feature alignment for multimodal disease localization in Chest X-Ray images

# Under updates
<img width="1621" height="808" alt="image" src="https://github.com/user-attachments/assets/9d26609f-5c4d-40ad-9a6e-d45808dc54f2" />

> [Investigating text-guided cross-region feature alignment for multimodal disease localization in Chest X-Ray images](https://www.authorea.com/doi/full/10.22541/au.175580188.86506576/v1)
> 
> Sourya Potti

## Installation 
Setup environment
```shell script
conda create --name cxrcodet python=3.8 -y
conda activate cxrcodet
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Under working directory 
git clone https://github.com/CVMI-Lab/CoDet.git
pip install ninja
pip install -v -U git+https://github.com/facebookresearch/xformers.git@7e05e2caaaf8060c1c6baadc2b04db02d5458a94
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..

#Install additional dependencies
cd CoDet/third_party/detectron2
pip install -e .
cd ../..
pip install -r requirements.txt
```

## Dataset Preparation
Please download the train images from the [VinDr dataset](https://www.kaggle.com/datasets/xhlulu/vinbigdata-chest-xray-resized-png-1024x1024) and move the images to the images folder, placing the data in the following way:
```
datasets/
        vindr/
            zero
            annotations/
            images/
                4d..92.jpg,
                ...
                
```

## Model Download
To download the cxr-codet model trained on VinDr, please download the [config](configs/CXRCoDet_VindrCXR_R50_1x) and the [model](https://drive.google.com/file/d/1UVM_WEbCUV8LggFbNRL_igYhr6nMLpe8/view?usp=sharing)

##






