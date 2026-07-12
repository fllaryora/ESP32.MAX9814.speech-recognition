#ifndef _MY_PCM_TO_SPRECTROGRAM_H
#define _MY_PCM_TO_SPRECTROGRAM_H

#include <kissfft/kiss_fft.h>
#include <kissfft/kiss_fft.c>
#include <kissfft/kiss_fftr.h>
#include <kissfft/kiss_fftr.c>
#include <kissfft/_kiss_fft_guts.h>

#define SAMPLES_FOR_EACH_FFT 320
#define SPECTRUM_BINS (SAMPLES_FOR_EACH_FFT/2 + 1)
#define AVG_POOLING_SIZE 4
#define EPSILON 1e-6f
#define POOLED_BINS_LENGTH ( (SPECTRUM_BINS + AVG_POOLING_SIZE - 1) / AVG_POOLING_SIZE)
#define FFT_STEP 160
#define AUDIO_LENGTH 48000

#define AMOUNT_OF_FRAMES_IN_OUTPUT  (1 + (AUDIO_LENGTH - SAMPLES_FOR_EACH_FFT) / FFT_STEP)
#define SPECTRUM_OUTPUT_SIZE POOLED_BINS_LENGTH * AMOUNT_OF_FRAMES_IN_OUTPUT

struct Spectrogram {
   int16_t * rawAudioDataInPcm;
   kiss_fftr_cfg kissFFTConfiguration;
   float pcmNormalizedWindow[SAMPLES_FOR_EACH_FFT];
   float hanningCoefficients[SAMPLES_FOR_EACH_FFT];
   kiss_fft_cpx fftOutput[SPECTRUM_BINS];
   bool firstFrame;
   float maxFrameEnergy;
   float meanFrameEnergy;

   float totalEnergy;
   float speechBandEnergy;

   float spectralFlux;
   float spectralFlatness;
    float previousSpectrum[POOLED_BINS_LENGTH];

   float* spectrogramOutput;
};

void initializeHanningWindows(Spectrogram * memory) {
const float argument = 2.0f * PI / (SAMPLES_FOR_EACH_FFT - 1.0f);

for (int sampleIndex = 0; sampleIndex < SAMPLES_FOR_EACH_FFT; sampleIndex++) {
   memory->hanningCoefficients[sampleIndex] =
  	0.5f - 0.5f * cosf(argument * sampleIndex);
}
}

void applyHanning(Spectrogram * memory) {
   for (int sampleIndex = 0; sampleIndex < SAMPLES_FOR_EACH_FFT; sampleIndex++) {
      memory->pcmNormalizedWindow[sampleIndex] *=
      memory->hanningCoefficients[sampleIndex];
   }
}

void getSpectrogramSegment( Spectrogram *memory, float *output){
    applyHanning(memory);

    kiss_fftr(
        memory->kissFFTConfiguration,
        memory->pcmNormalizedWindow,
        memory->fftOutput);

    memory->totalEnergy = 0.0f;
    memory->speechBandEnergy = 0.0f;
    memory->spectralFlux = 0.0f;

    float arithmeticMean = 0.0f;
    float geometricMean = 0.0f;

    int pooledIndex = 0;
    float pooledSum = 0.0f;
    int pooledCount = 0;

    for (int bin = 0; bin < SPECTRUM_BINS; bin++) {
        float real = memory->fftOutput[bin].r;
        float imag = memory->fftOutput[bin].i;

        float energy = real * real + imag * imag;

        //-------------------------------------------------
        // Statistics
        //-------------------------------------------------
        memory->totalEnergy += energy;
        arithmeticMean += energy;
        geometricMean += logf(energy + EPSILON);
        // human voice bins activity
        if (bin >= 6 && bin <= 68)
            memory->speechBandEnergy += energy;

        if (energy > memory->maxFrameEnergy)
            memory->maxFrameEnergy = energy;

        //-------------------------------------------------
        // Average Pooling
        //-------------------------------------------------
        pooledSum += energy;
        pooledCount++;

        bool lastBin = (bin == SPECTRUM_BINS - 1);

        if (pooledCount == AVG_POOLING_SIZE || lastBin) {
            float value = log10f( pooledSum / pooledCount + EPSILON);
            output[pooledIndex] = value;

            if (!memory->firstFrame) {
                memory->spectralFlux +=
                    fabsf( value - memory->previousSpectrum[pooledIndex]);
            }

            memory->previousSpectrum[ pooledIndex] = value;
            pooledIndex++;

            pooledSum = 0.0f;
            pooledCount = 0;
        }
    }

    arithmeticMean /= SPECTRUM_BINS;

    geometricMean =
        expf(
            geometricMean /
            SPECTRUM_BINS);

    memory->meanFrameEnergy =
        arithmeticMean;

    memory->spectralFlatness =
        geometricMean /
        (arithmeticMean + EPSILON);
}

bool get_spectrogram(
    Spectrogram *memory)
{
    memory->maxFrameEnergy = 0.0f;
    memory->firstFrame = true;

    for (int i = 0;
         i < POOLED_BINS_LENGTH;
         i++)
    {
        memory->previousSpectrum[i] = 0.0f;
    }

    //----------------------------------------------------
    // Mean
    //----------------------------------------------------

    float mean = 0.0f;

    for (int i = 0;
         i < AUDIO_LENGTH;
         i++)
    {
        mean +=
            memory->rawAudioDataInPcm[i];
    }

    mean /= AUDIO_LENGTH;

    //----------------------------------------------------
    // Normalization
    //----------------------------------------------------

    float maxDeviation = 0.0f;

    for (int i = 0;
         i < AUDIO_LENGTH;
         i++)
    {
        float deviation =
            fabsf(
                memory->rawAudioDataInPcm[i]
                - mean);

        if (deviation >
            maxDeviation)
        {
            maxDeviation =
                deviation;
        }
    }

    if (maxDeviation < EPSILON)
        maxDeviation = 1.0f;

    //----------------------------------------------------

    uint16_t currentRun = 0;
    uint16_t longestRun = 0;

    int frameIndex = 0;

    for (int sampleStart = 0;
         sampleStart +
         SAMPLES_FOR_EACH_FFT <=
         AUDIO_LENGTH;
         sampleStart += FFT_STEP)
    {
        //--------------------------------------------
        // Normalize PCM
        //--------------------------------------------

        for (int i = 0;
             i < SAMPLES_FOR_EACH_FFT;
             i++)
        {
            memory->pcmNormalizedWindow[i] =
                (
                    memory->rawAudioDataInPcm[
                        sampleStart + i]
                    - mean
                ) / maxDeviation;
        }

        //--------------------------------------------

        getSpectrogramSegment( memory, memory->spectrogramOutput + frameIndex * POOLED_BINS_LENGTH);

        //--------------------------------------------

        float speechRatio = memory->speechBandEnergy / (memory->totalEnergy + EPSILON);
        uint8_t score = 0;

        // Human voice use to be between  300 y 3400 Hz.
        if (speechRatio > 0.58f)
            score++;

        // The voice change since one frame to the next one
        if (memory->spectralFlux > 2.5f)
            score++;

        //
        // Voice ≈ 0.2..0.5
        // White noice ≈ 1
        //
        if (memory->spectralFlatness < 0.55f)
            score++;

        //
        // Only delete the total silence
        //
        if (memory->totalEnergy < 1e-4f)
            score = 0;

        bool voice = (score >= 2);

        if (voice) {
            currentRun++;

            if (currentRun > longestRun) {
                longestRun = currentRun;
            }
        }
        else {
            currentRun = 0;
        }

        memory->firstFrame = false;

        frameIndex++;
    }

    Serial.printf(
        "Ratio=%0.2f  Flux=%0.2f  Flat=%0.2f  Run=%u\n",
        memory->speechBandEnergy /
        (memory->totalEnergy + EPSILON),
        memory->spectralFlux,
        memory->spectralFlatness,
        longestRun);
        
    // voz por mas de 500ms
    return (longestRun >= 50);
}

#endif // _MY_PCM_TO_SPRECTROGRAM_H