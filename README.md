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

### Nsynth Dataset
Taken from the Nsynth [dataset](https://magenta.tensorflow.org/datasets/nsynth "NSyth dataset"), we analyzed 12 features that included both spectral and non-spectral features. Some spectral/temporal features include fast decay, long release, percussive, and reverb. Some specral features include brightness, darkness, and multiphonic.

### Neural Network Architecture and Performance

### 1) Binary classification: 

We first attempted to do binary classification across all instrument families. We thus had a simple multi-layer perceptron with 3 layers, having an input layer of 12 (relu activation) , hidden layer of 6 (relu activation), and output layer of 2 (sigmoid activation). Our loss function utilized sparse categorical cross entropy, compiled with an adam optimizer, and trained using 8 epochs and batch size of 12.

![binary](https://github.com/eglouberman/Instrument-AI/blob/master/Images/Binary_classification2.png)

As you can see, we achieved great results for neural network binary classification using simple neural network architecture.

### 2) Multi-class classification

Next, we modified our multi perceptron to hand multi-class. We thus had a simple multi-layer perceptron with 3 layers, having an input layer of 12 (relu activation) , hidden layer of 12 (relu activation), and output layer of 12 (softmax activation). Our loss function utilized categorical cross entropy, compiled with an adam optimizer, and trained using 8 epochs and batch size of 12.

![image](https://github.com/eglouberman/Instrument-AI/blob/master/Images/multiclassMLP.png)


We got a worse output accuracy for the multiclass model, which was expected due to the high amount of noise in the data. All 12 features only achieved around a 50% accuracy. Without percussive or reverb qualities, the model did not perform as well but still pretty similarly. This is important to note because percussive and reverb are temporal qualities, so this tells us that our system does not rely on utilizing temporal qualities too heavily.

# Convolutional Neural Network with Raw Audio as the input

We then decided to test whether we could only use spectral qualities from the audio data in order to see if a neural network can perform better. To do this, we decided to utilize a CNN because they have been empirically tested to work well with audio classification and they have a specific struture that mimic the biological structure of the human sensory system. They are organized in a series of preprocessing layers that perform different trasnforamtions and are great at detecting features/edges (Huang, 2018). 

We decided to play with different pre-processing methods and modify the input data to our network. In all, we compared utilizing an audio file's melspectrogram data, log(melspectrogram) data, and MFCC data. While all inputs did reasonabaly well, MFCC data yielded the best results, as shown below: 

![image](https://github.com/eglouberman/Instrument-AI/blob/master/Images/mfccs_confusion_with_title.png)

As you can see, our system achieved 94 percent accuracy overall. All instruments had a recall greater than 90 percent except for reed.


### Breaking the system and what it means for how the neural network behaves
In order to truly understand how the network functions, we tried to test our highly accurate model and see if it generalizes well to electronic and synthetic instruments. Given that it did well on acoustic instruments, we assumed that it should do a fairly good job at generalizing to other instruments since to the human ear, they sound similar. However, we achieved below 40 percent accuracy for both new instrument sources. This led us to believe that there was something intrinsically unique within the MFCC data to indicate a difference between acoustic, synthetic, and electronic sounds.

Additionally, we also tried to reverse the MFCC data to see if our model can still perform well. Theoretically, this should be easy since we attempted to take out all temporal effects. However, the system did not do a great job of generalizing in this case, as we achieved below a 40 percent accuracy. Thus, it led us to believe that our model did in fact take into account temporal qualities in the data to classify instrument family. Below is a chart that summarizes the breakdown of the model over different datasets.



<img align="middle" src="https://github.com/eglouberman/Instrument-AI/blob/master/Images/Instrument_types.png" alt="hi">




