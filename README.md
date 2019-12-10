# Instrument-AI

## Recognizing Instruments with Neural Networks
How well can a neural network perform when given the task to identify an instrument family when given raw audio data?  Using the NSynth dataset from Google, we achieved a 95% accuracy using a CNN that takes the Mel-Spectral Cepstral Coefficients (MFCC's) of the audio source. 

This project was completed as a final project for UCLA's Neural Networks class, Psych 186B. Trevor Harrison and I both worked on this project together. Neural Network architecture was inspired by an existing CNN audio project on [github](https://github.com/mikesmales).

## Background

What is sound in music? Musical instruments are unique to each other for several reasons. Primarily, instruments contain varying harmonic overtones, which make up its timbre. Note from different instruments are composed of unique combinations of harmonic frequencies. Sound can be identified by being deconstructed into its constituent frequences.

### Spectral importance

Spectral features are obtained by converting the time-based signal into the frequency domain using the Fourier Transform. Biological evidence demonstrates spectral feature detection ocurring in nature. The basilar membrane of the cochlea is tonotopically organized with one best frequency increasing the firing rate of a specific neuron. This process is similar to fourier analysis which breaks down a sound into harmonic frequencies. In audio pre-processing for machine learning and neural network tasks, this can be easily done using the Librosa library. Previous work have used spectral data to successfully classify timbre (David, 2005).

### Non-spectral importance

Since many neural network projects have succeeded with classification tasks using spectral features, we decided to first analyze how non-spectral features can also be utilized to accomplish the same task. Human beings are proven to recognize non-percussive instruments using the attack of its envelope. One study found that multi-layer perceptrons could identify instruments with a 93% accuracy and 80% accuracy only looking at attack (Agostini et al., 2003). Without attack, the system was only 71% accurate. This indicates non-spectral features are salient in instrument identification.

### Hypothesis
We decided to analyze whether spectral features by themselves are sufficient to classify the timbre of an instrument. Perhaps, a model with holistic information about a musical note's salient temporal, spectral and meta-features will be more effective at identifying its origin instrument thatn a model that only includes spectral features. 

### Nsynth [Dataset](link:https://magenta.tensorflow.org/datasets/nsynth "NSyth dataset")
We analyzed 13 features that included both spectral and non-spectral features.










