# AFPILD: Acoustic footstep dataset collected using one microphone array and LiDAR sensor for person identification and localization

This is a PyTorch implementation of our submitted [manuscript]().

**AFPILD:** Acoustic Footstep-based Person Identification and Localization Dataset.
**CRNN:** Convolutional Recurrent Neural Network.

## Acess to the AFPILD and this source code
**Note: The source code and the AFPILD are free for non-commercial research and education purposes.** Any commercial use should get formal permission first.
 
 The code and dataset are released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for NonCommercial use only. 

 
The AFPILD could be downloaded from [Baidu pan](https://pan.baidu.com/s/15UqRVKgxlz_CywCp6xy6DA) fetch code: ytvf or [Google Drive](https://drive.google.com/file/d/1FQMWbG8oAoZKXfLYsNZ_W_yLj9cZOUGu/view?usp=share_link).
Please cite our paper if you use any part of our source code or data in your research.


## Requirements
- python>=3.9.7
- audioread>=2.1.9
- PyTorch>=1.12.1
- torchvision>=0.13.1
- pandas>=1.4.4
- librosa>=0.8.1 
- h5py>=3.7.0
- numpy>=1.20.3
- scikit-learn>=1.0.1
- scipy>=1.7.3



## Usage 

Download the [AFPILD](https://drive.google.com/file/d/1FQMWbG8oAoZKXfLYsNZ_W_yLj9cZOUGu/view?usp=share_link) and prepare the directory following the below structure: 
```
├─ /AFPILD_v1_root
|   README.md				this file, markdown-format
|   LICENSE                 the license file
│   ├── S01
│   |    ├── s01_1_footstep_annotation.csv
│   |    ├── s01_1_footstep_audio.wav
│   |    ├── ...
│   |    ├── s01_4_footstep_annotation.csv
│   |    ├── s01_4_footstep_audio.wav
│   | 
│   ├── S02
│   |    ├── ...
...

│   ├── S40
│   |    ├── s40_1_footstep_annotation.csv
│   |    ├── s40_1_footstep_audio.wav
│   |    ├── ...
│   |    ├── s40_4_footstep_annotation.csv
│   |    ├── s40_4_footstep_audio.wav
```

1. Generate various dataset variants with files in ./scripts, including audio feature extraction: 
```
>> python ./spec_gcc_fea_ext_afpild.py 
```

2. Train and evaluate the model: 
```
The codebase is coming soon.
```


## Code References
In this Codebase, we utilize code from the following source(s):

* [wave-spec-fusion](https://github.com/denfed/wave-spec-fusion) 

* [CRNN](https://github.com/sharathadavanne/seld-dcase2022n) 

