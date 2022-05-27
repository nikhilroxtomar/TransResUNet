# TransResU-Net: Transformer based ResU-Net for Real-Time Colonoscopy Polyp Segmentation

## 1. Abstract
<div align="justify">
Colorectal cancer is one of the most common causes of cancer and cancer-related mortality worldwide. Colonoscopy is the primary technique to diagnose colon cancer. However, the polyp miss rate is significantly high. Early detection of polyp at the precancerous stage can help reduce the mortality rate. Deep learning based computer-aided diagnosis system may help colonoscopists to identify neoplastic polyps and act as a second pair of eyes. Additionally, CADX system could be a cost-effective solution and might contribute to long-term colorectal cancer prevention. In this study, we proposed deep learning-based architecture, Transformer ResU-Net (TransResU-Net), for automatic polyp segmentation. TresResU-Net is an encoder-decoder based architecture built upon residual block and takes the advantage of transformer self-attention mechanism and dilated convolution. Experimental result on two publicly available polyp segmentation benchmark datasets shows that TransResU-Net obtained a promising dice coefficient and a real-time speed.
</div>

## 2. Architecture
<img src="images/block-diagram.jpg">

## 3. Implementation
The proposed architecture is implemented using the PyTorch framework (1.9.0+cu111) with a single GeForce RTX 3090 GPU of 24 GB memory. 

### 3.1 Dataset
We have used the following datasets:
- [Kvasir-SEG](https://datasets.simula.no/downloads/kvasir-seg.zip)
- [BKAI](https://www.kaggle.com/competitions/bkai-igh-neopolyp/data)

BKAI dataset follows an 80:10:10 split for training, validation and testing, while the Kvasir-SEG follows an official split of 880/120.

### 3.2 Weight file
Updated soon.

## 4. Results

### 4.1 Quantative Results
Updated soon.

### 4.2 Qualitative Results
<img src="images/qualitative.jpg">

## 5. Citation
Updated soon.

## 6. License
The source code is free for research and education use only. Any comercial use should receive a formal permission from the first author.

## 7. Contact
Please contact nikhilroxtomar@gmail.com for any further questions. 
