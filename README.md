# Time-Frequency Domain Filter-and-Sum Network for Multi-channel Speech Separation

<!-- This repository provides the model implementation for the paper "Time-Frequency Domain Filter-and-Sum Network for Multi-channel Speech Separation". In this paper, we introduce a novel approach to multi-channel speech separation that improves upon implicit Filter-and-Sum Network (iFaSNet). Our approach involves transforming each module of the iFaSNet architecture to perform separation in the time-frequency domain.  The experimental results indicate that our proposed method is superior under the experimental conditions considered. -->

This repository contains the model implementation for the paper titled "Time-Frequency Domain Filter-and-Sum Network for Multi-channel Speech Separation." Our paper proposes a new approach to multi-channel speech separation, building upon the implicit Filter-and-Sum Network (iFaSNet). We achieve this by converting each module of the iFaSNet architecture to perform separation in the time-frequency domain. Our experimental results indicate that our method is superior under the considered conditions.

# Model

We implement the Time-Frequency Domain Filter-and-Sum Network (TF-FaSNet) based on iFaSNet's overall structure. The network performs end-to-end multi-channel speech separation in the time-frequency domain. Refer to the original paper for more information.

We propose the following improvements to enhance the performance of the iFaSNet model for separating mixtures:

- Use a multi-path separation module for spectral mapping in the T-F domain
- Add a 2D positional encoding to facilitate attention module learning spectro-temporal information
- Use narrow-band feature extraction to exploit inter-channel cues of different speakers
- Add a convolution module at the end of the separation module to capture local interactions and features.

The figure below shows the flowchart of TF-FaSNet model.

<p align="center">
    <img src="flowchart.png"  width="60%" height="30%">
</p>

# Usage

The implemention of the TF-FaSNet model can be found in **model.py**.

``` python
# test full model
mix_audio = torch.randn(3,6,64000)
test_model = make_TF_FasNet_4(
    nmic=6, nspk=2, n_fft=256, 
    D=16, B=4, I=8, J=1, H=128, E=4, L=4
    )
separated_audio = test_model(mix_audio)
```

