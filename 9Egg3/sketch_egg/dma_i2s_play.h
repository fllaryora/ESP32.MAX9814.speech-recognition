#ifndef _MY_DMA_I2S_PLAY_H
#define _MY_DMA_I2S_PLAY_H

#include <math.h>
#include "driver/i2s.h"
#include "./dma_i2s_record.h"

/*
 * MAX98357A wiring (left DevKit header, same side as mic VP):
 *
 *   GPIO 14  -->  LRC   (LRCLK / WS)
 *   GPIO 27  -->  BCLK
 *   GPIO 26  -->  DIN
 *
 *   GAIN     -->  100 kΩ --> GND     (15 dB)
 *   SD       -->  Float (leave unconnected)  // left-channel mono
 *   GND      -->  GND
 *   Vin      -->  3.3 V
 *   Speaker+ -->  amp Out+
 *   Speaker- -->  amp Out-
 */

#define I2S_SPEAKER_PORT I2S_NUM_1
#define I2S_LRC_PIN      14
#define I2S_BCLK_PIN     27
#define I2S_DOUT_PIN     26

#define PLAY_SINE_HZ           440
#define PLAY_SINE_AMPLITUDE    10000
#define PLAY_SINE_DURATION_MS  1000
#define PLAY_CHUNK_SAMPLES     256

void stopMicForPlayback() {
  teardownI2S();
}

void restoreMicI2S() {
  setupI2S();
}

void setupSpeakerI2S() {
  static const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
  };

  esp_err_t installStatus = i2s_driver_install(I2S_SPEAKER_PORT, &i2s_config, 0, NULL);
  if (ESP_OK != installStatus) {
    Serial.println(" Error : speaker i2s_driver_install failed");
    while (true);
  }

  static const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCLK_PIN,
    .ws_io_num = I2S_LRC_PIN,
    .data_out_num = I2S_DOUT_PIN,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  esp_err_t pinStatus = i2s_set_pin(I2S_SPEAKER_PORT, &pin_config);
  if (ESP_OK != pinStatus) {
    Serial.println(" Error : speaker i2s_set_pin failed");
    while (true);
  }
}

void teardownSpeakerI2S() {
  i2s_driver_uninstall(I2S_SPEAKER_PORT);
}

void playPcm16(const int16_t *samples, size_t sampleCount, uint8_t errorBlinkLed) {
  size_t bytesToWrite = sampleCount * sizeof(int16_t);
  const uint8_t *head = (const uint8_t *)samples;
  size_t totalWritten = 0;

  while (totalWritten < bytesToWrite) {
    size_t bytesWritten = 0;
    esp_err_t status = i2s_write(
      I2S_SPEAKER_PORT,
      head + totalWritten,
      bytesToWrite - totalWritten,
      &bytesWritten,
      DO_NOT_FORSAKE_IN_CASE_OF_TIME_OUT
    );
    if (ESP_OK != status) {
      Serial.println(" Error : i2s_write failed");
      while (true) {
        digitalWrite(errorBlinkLed, LOW);
        delay(500);
        digitalWrite(errorBlinkLed, HIGH);
        delay(500);
      }
    }
    totalWritten += bytesWritten;
  }
}

/* Fill buffer with a sine chunk starting at sampleIndex; returns samples written. */
size_t fillSineChunk(int16_t *buffer, size_t capacity, size_t sampleIndex, size_t totalSamples) {
  size_t n = capacity;
  if (sampleIndex + n > totalSamples) {
    n = totalSamples - sampleIndex;
  }
  const float twoPiF = 2.0f * (float)M_PI * (float)PLAY_SINE_HZ / (float)SAMPLE_RATE;
  for (size_t i = 0; i < n; i++) {
    float phase = twoPiF * (float)(sampleIndex + i);
    buffer[i] = (int16_t)(sinf(phase) * (float)PLAY_SINE_AMPLITUDE);
  }
  return n;
}

/* Half-duplex: stop mic, play ~1 s 440 Hz tone, restore mic. */
void playTestSineBeep(uint8_t errorBlinkLed) {
  const size_t totalSamples = (size_t)SAMPLE_RATE * (size_t)PLAY_SINE_DURATION_MS / 1000u;
  int16_t chunk[PLAY_CHUNK_SAMPLES];

  Serial.println("PLAY");
  stopMicForPlayback();
  setupSpeakerI2S();

  size_t sampleIndex = 0;
  while (sampleIndex < totalSamples) {
    size_t n = fillSineChunk(chunk, PLAY_CHUNK_SAMPLES, sampleIndex, totalSamples);
    playPcm16(chunk, n, errorBlinkLed);
    sampleIndex += n;
  }

  // Let DMA drain the last buffers before uninstall.
  delay(50);
  teardownSpeakerI2S();
  restoreMicI2S();
  Serial.println("PLAY_DONE");
}

#endif // _MY_DMA_I2S_PLAY_H
