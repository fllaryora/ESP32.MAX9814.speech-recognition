#ifndef _MY_PCM_TO_SPRECTROGRAM_H   
#define _MY_PCM_TO_SPRECTROGRAM_H   

#include <kissfft/kiss_fft.h>
#include <kissfft/kiss_fft.c>
#include <kissfft/kiss_fftr.h>
#include <kissfft/kiss_fftr.c>
#include <kissfft/_kiss_fft_guts.h>
#define SAMPLES_FOR_EACH_FFT 320
#define SPECTRUM_BINS (SAMPLES_FOR_EACH_FFT/2 + 1) //161
// AVG pooling: every 4 output bins are averaged into a single bin. 	
#define AVG_POOLING_SIZE 4
#define EPSILON 1e-6f 
// POOLED_BINS == ceil( SPECTRUM_BINS / AVG_POOLING_SIZE) 
#define POOLED_BINS_LENGTH ( (SPECTRUM_BINS + AVG_POOLING_SIZE - 1) / AVG_POOLING_SIZE) //41 
// Hop size between the neighbor windows.
#define FFT_STEP 160
#define AUDIO_LENGTH 48000	// Longitud de la señal de audio en muestras: 16000 corresponde a un segundo a 16 kHz.

#define AMOUNT_OF_FRAMES_IN_OUTPUT  (1 + (AUDIO_LENGTH - SAMPLES_FOR_EACH_FFT) / FFT_STEP) //299
#define SPECTRUM_OUTPUT_SIZE POOLED_BINS_LENGTH * AMOUNT_OF_FRAMES_IN_OUTPUT //12259

struct Spectrogram {
  int16_t * rawAudioDataInPcm; // size AUDIO_LENGTH
  kiss_fftr_cfg kissFFTConfiguration; // set to NULL on construction
  float pcmNormalizedWindow[SAMPLES_FOR_EACH_FFT];
  float  hanningCoefficients[SAMPLES_FOR_EACH_FFT];
  kiss_fft_cpx fftOutput[SPECTRUM_BINS];
  float fftEnergy[SPECTRUM_BINS];
  float smoothedNoiseFloor; // set to 0.0f on construction
  float*  spectrogramOutput; // size SPECTRUM_OUTPUT_SIZE;
};

/*
The standard configuration of MFCC is windows == 20 ms, hop == 10ms.

I am going to use windows == 25 ms, hop == 10ms.

SAMPLES_FOR_EACH_FFT == 320
Each fft call will consume 320 samples of audio.
320 samples / 16kHz = 20 mili sec.


There is 50Hz per BIN

The Hop size between the neighbor windows.
The hop size refers to how many samples I need to skip to start the next FFT.
The windows overlap by 50%.
320 / 2 = 160

160 samples / 16kHz = 10 mili sec.


It means It will call of fft 299 times for 3 seconds of 16000khz.
the last sample for the first time is: 320.
the last sample for the second time is: 480.
(last_sample) = 320 + 160 (times-1)
If the last sample is 48000, I get (times-1)== 298 

 AVG_POOLING_SIZE == Number of frequency bins that are averaged (pooled) together to reduce the size of the spectrogram.
 SPECTRUM_BINS == Number of unique frequency bins in the FFT for a window of length SAMPLES_FOR_EACH_FFT (161 bins before pooling).

  POOLED_BINS == Number of FFT frequency bins after average pooling (~41 bins).

  EPSILON == Small constant for numerical stability (to avoid division by zero and log(0)) during subsequent processing.

  Now letme explain a little math:
  Ceil-operation takes a real number and rounds it up to the smallest integer that is greater than or equal to that number.
  Floor-operation rounds a real number down to the nearest integer that is less than or equal to that number.
  Identity of math said: Demostration in: https://en.wikipedia.org/wiki/Floor_and_ceiling_functions
  ceil( a/b) == floor ( (a+b-1)/b )

  ceil( SPECTRUM_BINS / AVG_POOLING_SIZE) == floor ( (SPECTRUM_BINS + AVG_POOLING_SIZE -1) / AVG_POOLING_SIZE )

  The spectrogram has a frame size of 41 pixels => 164 bytes.
  Each element of the spectrogram is a frame.
  299 frames in the out.
  So 299 * 41=  12259 pixels => 49036 bytes.
  With 32 bits of address
  PLUS the double pointer with 299 elements => 299 * 4 bytes.
  Total of each spectrogram = 50232 bytes. 
*/

///
// ===============================
// Initialize the hanning window coefficients.
// ===============================
void initializeHanningWindows(Spectrogram * memory) {
  const float argument = 2.0f * PI / (SAMPLES_FOR_EACH_FFT - 1.0f);
  for (int sampleIndex = 0; sampleIndex < SAMPLES_FOR_EACH_FFT; sampleIndex++) {
	  memory->hanningCoefficients[sampleIndex] = 0.5f - 0.5f * cosf(argument * sampleIndex);
  }
}

// ===============================
// It applys the hanning window to a window sample
// ===============================
void applyHanning(Spectrogram * memory) {
  for (int sampleIndex = 0; sampleIndex < SAMPLES_FOR_EACH_FFT; sampleIndex++) {
	  memory->pcmNormalizedWindow[sampleIndex] *= memory->hanningCoefficients[sampleIndex];
  }
}

// ===============================
// This function takes the normalizedPCM and apply the hanning wrapper wave on it (in-place) SAMPLES_FOR_EACH_FFT times.

// After that appy ONCE the fft for SAMPLES_FOR_EACH_FFT samples the output is stored in fftOutput.
// fftOutput has the real and the imaginary part of the signal.
// Why is `SPECTRUM_BINS` equal to half the total number of samples plus one for the FFT?
// This is because the FFT input in the time domain consists of 320 samples,
// and the FFT output consists of 320 bins in the frequency domain:
// first the negative frequency bins, then the zero bin, followed by the positive frequency bins.
// Since the output is always mirrored, the negative-frequency portion is discarded. 

// Then It takes the square of the module and store it in fftEnergy

// Then reduce the spectum segment with average pooling but store in output the log base 10 of the average.

// load in the output pointer a segment of the spectrogram (a window) using average pooling
//  - float *normalizedPCM: (temporal domain)Input data for the Discrete Fourier Transform (DFT/FFT) (320 samples).
//  - float *output: Frequency-amplitude response calculated for the input sample.
// =============================== 

void getSpectrogramSegment(Spectrogram * memory, float *output) {
  
  applyHanning(memory);
 
  kiss_fftr(memory->kissFFTConfiguration, memory->pcmNormalizedWindow, memory->fftOutput);

  for (int binIndex = 0; binIndex < SPECTRUM_BINS; binIndex++) {
	  float realPart = memory->fftOutput[binIndex].r;
	  float imaginaryPart = memory->fftOutput[binIndex].i;
	  memory->fftEnergy[binIndex] = realPart * realPart + imaginaryPart * imaginaryPart;
  }
  
  int outputIndex = 0;

  // Average pooling stage
  for (int pooledIndex = 0;
      pooledIndex < POOLED_BINS_LENGTH;
      pooledIndex++)
  {
      int firstBin = pooledIndex * AVG_POOLING_SIZE;

      float sum = 0.0f;
      int validBin = 0;

      for (int offset = 0; offset < AVG_POOLING_SIZE; offset++)
      {
          int bin = firstBin + offset;

          if (bin < SPECTRUM_BINS){
            sum += memory->fftEnergy[bin];
            validBin++;
          }

      }

      output[pooledIndex] =
          log10f(sum / validBin + EPSILON);
  }

}


// ===============================
// Main function for constructing the spectrogram with noise level detection.
// Returns true if the sound level exceeds the noise level.
//
// ===============================
bool get_spectrogram(  Spectrogram * memory ){

  //=== Second stage == from pcm  Voice activity detector
  float mean = 0.0f;
  for (size_t sampleIndex = 0; sampleIndex < AUDIO_LENGTH; sampleIndex++) {
	  mean += memory->rawAudioDataInPcm[sampleIndex];
  }

  mean /= AUDIO_LENGTH;
 
  // Maximum deviation from the average
  float maxDeviationFromAverage = fabsf((float)memory->rawAudioDataInPcm[0] - mean);
  //Average Absolute Deviation: It measures how far, on average,
  // the samples deviate from the mean, but without squaring the differences.
  float noiseFloor = 0.0f;
  int samplesOverNoiseFloor = 0;

  for (size_t sampleIndex = 0; sampleIndex < AUDIO_LENGTH; sampleIndex++) {

    float deviationFromAverage = fabsf((float)memory->rawAudioDataInPcm[sampleIndex] - mean);

    maxDeviationFromAverage = max(maxDeviationFromAverage, deviationFromAverage);

    // Acumular la suma de desviaciones absolutas.
    noiseFloor += deviationFromAverage;
    
    if (deviationFromAverage > 5.0f * memory->smoothedNoiseFloor) {
      samplesOverNoiseFloor++;
    }
  }
  noiseFloor /= AUDIO_LENGTH;
 

  if (noiseFloor < memory->smoothedNoiseFloor) {
	  memory->smoothedNoiseFloor = 0.7f * memory->smoothedNoiseFloor + 0.3f * noiseFloor;
  } else {
  	memory->smoothedNoiseFloor = 0.99f * memory->smoothedNoiseFloor + 0.01f * noiseFloor;
  }
 
  // avoid division by 0
  if (maxDeviationFromAverage < EPSILON) {
	  maxDeviationFromAverage = 1.0f;
  }

  int frameIndex = 0;
  //float pcmNormalizedWindow[SAMPLES_FOR_EACH_FFT];
  for (size_t sampleIndexStart = 0; sampleIndexStart + SAMPLES_FOR_EACH_FFT <= AUDIO_LENGTH; sampleIndexStart += FFT_STEP) {
	  //=== Third stage == from pcm  to pcm normalized
    for (int sampleIndexOffset = 0; sampleIndexOffset < SAMPLES_FOR_EACH_FFT; sampleIndexOffset++) {
      memory->pcmNormalizedWindow[sampleIndexOffset] =
         ( (float) memory->rawAudioDataInPcm[sampleIndexStart + sampleIndexOffset] - mean ) / maxDeviationFromAverage;
    }
    //=== Fourth stage == from pcm normalized to single frame of spectrogram
    getSpectrogramSegment(memory, memory->spectrogramOutput + (frameIndex * POOLED_BINS_LENGTH));
    frameIndex++;

  	if (frameIndex >= AMOUNT_OF_FRAMES_IN_OUTPUT) break;
  }
 
  // 5% is 5/100 = 0.05
  // and 1/20 = 0.05
  bool isSamplesAboveNoise = samplesOverNoiseFloor > (AUDIO_LENGTH / 20);
 
  return isSamplesAboveNoise;
}

#endif // _MY_PCM_TO_SPRECTROGRAM_H   
