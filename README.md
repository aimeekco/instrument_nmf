# NMF for Musical Feature Learning
We intend to explore to what extent Nonnegative Matrix Factorization (NMF) can effectively separate musical instruments from mixtures using the MUSDB18-HQ dataset. Specifically, we compare two variants of NMF based on different divergence measures: Kullback-Liebler (KL) and Itakura-Saito (IS), evaluating how their underlying assumptions about noise affect source separation in music. As an extension, we implement a semi-supervised NMF (SSNMF) model, in which instrument-specific bases are pretrained on isolated stems and then fixed during inference on mixed tracks. Performance will be quantitatively measured using signal-to-distortion (SDR) and signal-to-interference (SIR) metrics, and qualitatively through spectrogram visualization. Ultimately, our goal is to understand how simple additive models can capture the structure of audio mixtures, and whether partial supervision improves separation without relying on deep learning methods that dominate the modern signal separation landscape. 

## Notes
Download MUSDB18-HQ from https://zenodo.org/records/3338373 and put folder in dataset/

## MUSDB18-HQ Dataset:
Rafii, Zafar, Antoine Liutkus, Fabian-Robert Stöter, Stylianos Ioannis Mimilakis, and Rachel Bittner. “MUSDB18-HQ - an Uncompressed Version of MUSDB18”. Zenodo, August 1, 2019. https://doi.org/10.5281/zenodo.3338373.
