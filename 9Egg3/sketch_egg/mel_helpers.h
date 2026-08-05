#ifndef _MEL_HELPERS_H
#define _MEL_HELPERS_H
#include <math.h>
#include <stdint.h>

#define MIN_FRECUENCY 0.0f
#define MAX_FRECUENCY 8000.0f
//getMelFrecuency (MIN_FRECUENCY)
#define MIN_MEL_FRECUENCY 0.0f
//getMelFrecuency (MAX_FRECUENCY)
#define MAX_MEL_FRECUENCY 2840.023047f
//(MAX_MEL_FRECUENCY - MIN_MEL_FRECUENCY) / (MEL_DOTS + 1) = 202.8587891f
#define MEL_WIDTH_BETWEEN_BINS 202.8587891f

// 320 samples for each FFT
// 16000 samplerate
// 16000 / 320 = 50 delta frecuency
#define VANILLA_WIDTH_BETWEEN_BINS 50.0f
//getMelFrecuency (VANILLA_WIDTH_BETWEEN_BINS)
#define DELTA_FRECUENCY_MELS 77.75456466f

// (MAX_MEL_FRECUENCY - MIN_MEL_FRECUENCY) / (MEL_WIDTH_BETWEEN_BINS- MIN_MEL_FRECUENCY) -1
#define LIMIT_MAX_MEL_DOTS 35.52548322f
// floor of LIMIT_MAX_MEL_DOTS
#define FLOOR_LIMIT_MAX_MEL_DOTS 35

#define MEL_DOTS 13
// assert MEL_DOTS is less than FLOOR_LIMIT_MAX_MEL_DOTS

struct TriangularFilter {
    uint16_t left;
    uint16_t center;
    uint16_t right;
};

// Convert Hz to Mel
float getMelFrecuency(float vanillaFrecuency) {
    return 2595.0f * log10f(1 + (vanillaFrecuency) / 700.0f);
}

// Convert Mel to Hz
float getVanillaFrecuency(float melFrecuency) {
    return 700.0f * (powf(10.0f, (melFrecuency / 2595.0f)) - 1.0f);
}

/*
    It returns the mel frecuency from the filter number
*/
float getMelFrecuencyFromFilterNumber(uint16_t melFilterBankNumber) {
    return MIN_MEL_FRECUENCY + melFilterBankNumber * MEL_WIDTH_BETWEEN_BINS;
}

/*
# It return the bin index from the filter number
*/
uint16_t getBoundFromFilterNumber(uint16_t melFilterBankNumber) {
    float melFrecuency = getMelFrecuencyFromFilterNumber(melFilterBankNumber);
    float vanillaFrecuency = getVanillaFrecuency(melFrecuency);
    return (uint16_t)(vanillaFrecuency / VANILLA_WIDTH_BETWEEN_BINS);
}

/*
* It returns the left, center and right bounds of the filter bank index
*/
TriangularFilter getMelFilterBankBounds(uint16_t melFilterBankNumber) {
    TriangularFilter filter;
    filter.left = getBoundFromFilterNumber(melFilterBankNumber);
    filter.center = getBoundFromFilterNumber(melFilterBankNumber + 1);
    filter.right = getBoundFromFilterNumber(melFilterBankNumber + 2);
    return filter;
}

/*
# It returns the triangular filter weight for the bin number in the mel spectrogram
*/
float triangularFilterH(uint16_t binNumberInMelSpectrogram, TriangularFilter filter) {
    if (binNumberInMelSpectrogram < filter.left || binNumberInMelSpectrogram > filter.right) {
        return 0.0f;
    }
    float numerator = 0.0f;
    float denominator = 0.0f;
    if (binNumberInMelSpectrogram <= filter.center) {
        numerator = (float)(binNumberInMelSpectrogram - filter.left);
        denominator = (float)(filter.center - filter.left);
    } else {
        numerator = (float)(filter.right - binNumberInMelSpectrogram);
        denominator = (float)(filter.right - filter.center);
    }
    return numerator / denominator;
}

#endif // _MEL_HELPERS_H
