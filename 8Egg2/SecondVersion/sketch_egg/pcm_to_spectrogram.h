#ifndef _MY_PCM_TO_SPRECTROGRAM_H
#define _MY_PCM_TO_SPRECTROGRAM_H

#include <kissfft/kiss_fft.h>
#include <kissfft/kiss_fft.c>
#include <kissfft/kiss_fftr.h>
#include <kissfft/kiss_fftr.c>
#include <kissfft/_kiss_fft_guts.h>

#include "./mel_helpers.h"

#define SAMPLES_FOR_EACH_FFT 320
#define SPECTRUM_BINS (SAMPLES_FOR_EACH_FFT/2 + 1) //161 bins
#define AVG_POOLING_SIZE 4 //4 bins
#define EPSILON 1e-6f


#define FFT_STEP 160 //160 samples
#define AUDIO_LENGTH 24000// == TOTAL_SAMPLES from the other h ex 48000

#define AMOUNT_OF_FRAMES_IN_OUTPUT  (1 + (AUDIO_LENGTH - SAMPLES_FOR_EACH_FFT) / FFT_STEP) //149 frames

// 13 bins * 149 frames = 1937 bins
#define SPECTRUM_OUTPUT_SIZE (MEL_DOTS * AMOUNT_OF_FRAMES_IN_OUTPUT)

//---- VAD (AGC-friendly, mel-based) ---------------------------------
// First frames estimate ambient noise (relative to this clip).
#define VAD_NOISE_FRAMES 25          // ~250 ms at 10 ms hop
// Low-mid mel bands ≈ speech formants for 13 mels / 0-8 kHz
#define VAD_SPEECH_MEL_END 9         // bands [0 .. 8]
#define VAD_EXCESS_THRESHOLD 0.12f   // mean(mel - noiseMel)
#define VAD_FLUX_THRESHOLD 0.06f     // mean |Δmel|
#define VAD_SPEECH_RATIO_THRESHOLD 0.55f
#define VAD_MIN_SCORE 2
#define VAD_MIN_RUN 40               // ~400 ms contiguous speech

struct Spectrogram {
   int16_t * rawAudioDataInPcm;
   kiss_fftr_cfg kissFFTConfiguration;
   float pcmNormalizedWindow[SAMPLES_FOR_EACH_FFT];
   float hanningCoefficients[SAMPLES_FOR_EACH_FFT];
   kiss_fft_cpx fftOutput[SPECTRUM_BINS];

   bool firstFrame;

   // Used by VAD spectral flux (keep this)
   float previousFrame[MEL_DOTS];
   float noiseMel[MEL_DOTS];

   float* melSpectrogramOutput;
};

void applyHanning(Spectrogram * memory);

/*
    One frame Size:
    previousFrame represents the previous frame of the spectrogram
*/
inline void setZeroPreviousSpectrumFrame(Spectrogram * memory) {
    for (int binIndex = 0; binIndex < MEL_DOTS; binIndex++) {
        memory->previousFrame[binIndex] = 0.0f;
    }
}

void initializeHanningWindows(Spectrogram * memory) {
    const float argument = 2.0f * PI / (SAMPLES_FOR_EACH_FFT - 1.0f);

    for (int sampleIndex = 0; sampleIndex < SAMPLES_FOR_EACH_FFT; sampleIndex++) {
        memory->hanningCoefficients[sampleIndex] =
            0.5f - 0.5f * cosf(argument * sampleIndex);
    }
}

/*
    From Raw audio data, create a normalized frame of audio
    and apply the Hanning window
    pcmNormalizedWindow: size 320
*/
inline void prepareFrameOfAudio( Spectrogram * memory,
     float mean, float maxDeviation, int sampleStart) {
    // Normalize the frame of audio
    for (int sampleIndex = 0; sampleIndex < SAMPLES_FOR_EACH_FFT; sampleIndex++) {
       memory->pcmNormalizedWindow[sampleIndex] =
           (memory->rawAudioDataInPcm[sampleStart + sampleIndex] - mean) / maxDeviation;
    }
    applyHanning(memory);
}

/*
    On a normalized Frame of audio, apply the Hanning window
    Input: normalized frame of audio
    Output: windowed frame of audio
*/
void applyHanning(Spectrogram * memory) {
   for (int sampleIndex = 0; sampleIndex < SAMPLES_FOR_EACH_FFT; sampleIndex++) {
      memory->pcmNormalizedWindow[sampleIndex] *=
        memory->hanningCoefficients[sampleIndex];
   }
}

//----------------------------------------------------
// Mean of the whole audio
//----------------------------------------------------
float getMeanOfAudio(Spectrogram * memory) {
    float mean = 0.0f;
    for (int sampleIndex = 0; sampleIndex < AUDIO_LENGTH; sampleIndex++){
        mean += memory->rawAudioDataInPcm[sampleIndex];
    }
    mean /= AUDIO_LENGTH;
    return mean;
}

//----------------------------------------------------
// Max deviation of the whole audio
//----------------------------------------------------
float getMaxDeviationOfAudio(Spectrogram * memory, float mean) {
    float maxDeviation = 0.0f;

    for (int sampleIndex = 0; sampleIndex < AUDIO_LENGTH; sampleIndex++)
    {
        float deviation =
            fabsf(memory->rawAudioDataInPcm[sampleIndex] - mean);

        if (deviation > maxDeviation) {
            maxDeviation = deviation;
        }
    }

    return maxDeviation;
}

/*
Porcess a single frame of the spectrogram
getSpectrogramSegment:
- kiss_fftr: pcmNormalizedWindow -> fftOutput : size 161
- mel filter bank -> melSpectrogramOutput[frame]

input:  pcmNormalizedWindow (normalized and windowed frame of audio)
output: mel frame written; previousFrame updated for flux
*/
void getSpectrogramSegment( Spectrogram * memory, int frameIndex) {
    float * output = memory->melSpectrogramOutput + frameIndex * MEL_DOTS;

    kiss_fftr(
        memory->kissFFTConfiguration,
        memory->pcmNormalizedWindow,
        memory->fftOutput);

    float * temporalLogEnergyFrame = memory->pcmNormalizedWindow;

    for (int bin = 0; bin < SPECTRUM_BINS; bin++) {
        float real = memory->fftOutput[bin].r;
        float imag = memory->fftOutput[bin].i;

        // module of bin
        float energy = log10f( sqrtf( real * real + imag * imag ) + EPSILON);
        temporalLogEnergyFrame[bin] = energy;
    }

    for (int melFilterBankIndex = 0; melFilterBankIndex < MEL_DOTS; melFilterBankIndex++) {
        TriangularFilter filter = getMelFilterBankBounds(melFilterBankIndex);
        float filterValue = 0.0f;
        for (int bin = (filter.left+1) ; bin <= (filter.right-1); bin++) {
            filterValue += temporalLogEnergyFrame[bin] * triangularFilterH(bin, filter);
        }
        output[melFilterBankIndex] = filterValue;
    }
}

/*
    Score one mel frame vs noiseMel / previousFrame.
    Returns true if frame looks like speech (vote >= VAD_MIN_SCORE).
*/
bool scoreMelFrameAsVoice(Spectrogram * memory, const float * mel) {
    float excessSum = 0.0f;
    float fluxSum = 0.0f;
    float speechEnergy = 0.0f;
    float totalEnergy = 0.0f;

    for (int k = 0; k < MEL_DOTS; k++) {
        excessSum += mel[k] - memory->noiseMel[k];
        fluxSum += fabsf(mel[k] - memory->previousFrame[k]);

        // Shift log-mel up so ratio uses positive mass
        float positive = mel[k] - memory->noiseMel[k];
        if (positive < 0.0f) {
            positive = 0.0f;
        }
        totalEnergy += positive;
        if (k < VAD_SPEECH_MEL_END) {
            speechEnergy += positive;
        }
    }

    float excess = excessSum / (float)MEL_DOTS;
    float flux = fluxSum / (float)MEL_DOTS;
    float speechRatio = speechEnergy / (totalEnergy + EPSILON);

    uint8_t score = 0;
    if (excess > VAD_EXCESS_THRESHOLD) {
        score++;
    }
    if (!memory->firstFrame && flux > VAD_FLUX_THRESHOLD) {
        score++;
    }
    if (speechRatio > VAD_SPEECH_RATIO_THRESHOLD) {
        score++;
    }

    return score >= VAD_MIN_SCORE;
}

/*
get_spectrogram:
- prepareFrameOfAudio: rawAudioDataInPcm -> pcmNormalizedWindow
- getSpectrogramSegment: -> melSpectrogramOutput
- mel VAD with noise floor + flux (previousFrame) + speech-band ratio
*/
bool get_spectrogram( Spectrogram * memory) {
    memory->firstFrame = true;

    setZeroPreviousSpectrumFrame(memory);
    float mean = getMeanOfAudio(memory);
    float maxDeviation = getMaxDeviationOfAudio(memory, mean);

    // Digital silence / empty buffer after AGC path
    if (maxDeviation < EPSILON) {
        return false;
    }

    for (int k = 0; k < MEL_DOTS; k++) {
        memory->noiseMel[k] = 0.0f;
    }

    uint16_t currentRun = 0;
    uint16_t longestRun = 0;
    int frameIndex = 0;

    for (int sampleStart = 0;
         sampleStart + SAMPLES_FOR_EACH_FFT <= AUDIO_LENGTH;
         sampleStart += FFT_STEP) {

        prepareFrameOfAudio(memory, mean, maxDeviation, sampleStart);
        getSpectrogramSegment(memory, frameIndex);

        float * mel = memory->melSpectrogramOutput + frameIndex * MEL_DOTS;

        if (frameIndex < VAD_NOISE_FRAMES) {
            for (int k = 0; k < MEL_DOTS; k++) {
                memory->noiseMel[k] += mel[k];
            }
        } else {
            if (frameIndex == VAD_NOISE_FRAMES) {
                for (int k = 0; k < MEL_DOTS; k++) {
                    memory->noiseMel[k] /= (float)VAD_NOISE_FRAMES;
                }
            }

            bool voice = scoreMelFrameAsVoice(memory, mel);
            if (voice) {
                currentRun++;
                if (currentRun > longestRun) {
                    longestRun = currentRun;
                }
            } else {
                currentRun = 0;
            }
        }

        for (int k = 0; k < MEL_DOTS; k++) {
            memory->previousFrame[k] = mel[k];
        }
        memory->firstFrame = false;
        frameIndex++;
    }

    return (longestRun >= VAD_MIN_RUN);
}

#endif // _MY_PCM_TO_SPRECTROGRAM_H
