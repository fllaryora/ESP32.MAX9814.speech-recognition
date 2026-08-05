#ifndef _MY_DMA_I2S_PLAY_H
#define _MY_DMA_I2S_PLAY_H

#include <math.h>
#include "driver/i2s.h"
#include "./dma_i2s_record.h"

#ifndef PLAY_TEST
#define PLAY_TEST 0
#endif

/*
 * MAX98357A wiring (left DevKit header, same side as mic VP):
 *
 *   GPIO 14  -->  LRC   (LRCLK / WS)
 *   GPIO 27  -->  BCLK
 *   GPIO 26  -->  DIN
 *
 *   GAIN     -->  pull down 100 kΩ --> GND     (15 dB)
 *   GAIN     -->  GND     (12 dB)  (*use it)
 *   GAIN     -->  Floating     (9 dB)
 *   GAIN     -->  Vin     (6 dB)
 *   GAIN     -->  pull up 100 kΩ --> Vin (3 dB)
 *
 *   SD       -->  (< 0.16 V) GND  // SHUT DOWN 
 *   SD       -->  (0.16 V- 0.77V) Float (leave unconnected)  // (LEFT + RIGHT)/2
 *   SD       -->  (0.77 V - 1.4V) pull up 39 kΩ --> Vin     (RIGHT)
 *   SD       -->  (> 1.4 V) VIN   (LEFT). (*use it)

 *   GND      -->  GND
 *   Vin      -->  3.3 V (0.6 Watts)
 *   Speaker+ -->  amp Out+
 *   Speaker- -->  amp Out-
 */

#define I2S_SPEAKER_PORT I2S_NUM_1 //  refers to the second physical I2S hardware controller inside the ESP32 chip.
#define I2S_LRC_PIN      14
#define I2S_BCLK_PIN     27
#define I2S_DOUT_PIN     26

#define PLAY_SINE_HZ           440
#define PLAY_SINE_AMPLITUDE    10000
#define PLAY_SINE_DURATION_MS  1000
#define PLAY_CHUNK_SAMPLES     256

/* Standard MPEG-1 Layer III decode target (consumer MP3). */
#define MP3_SAMPLE_RATE        44100
/* MPEG-1 Layer III: 1152 PCM samples per channel per frame. */
#define MP3_FRAME_SAMPLES      1152

//Esto llama al #include "./dma_i2s_record.h"
void stopMicForPlayback() {
  teardownI2S();
}

//Esto llama al #include "./dma_i2s_record.h"
void restoreMicI2S() {
  setupI2S();
}

void setupSpeakerI2S() {

#if PLAY_TEST
  static const i2s_config_t i2s_config = {
    // speaker mode
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SAMPLE_RATE,
    // 16 bits mode - Sound like a shit but it is ok for me?? 
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    // ONLY LEFT (SD --> VIN)
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    // Philips mode 
    .communication_format = I2S_COMM_FORMAT_I2S,
    // interruption flags///< Accept a Level 1 interrupt vector (lowest priority)
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
  };
#else
  /* PCM after standard MP3 decode: 44.1 kHz, 16-bit, stereo interleaved (LRLR...).
   * SD --> VIN selects LEFT word only from that stereo stream.
   * dma_buf_len sized to one MPEG-1 Layer III frame; APLL for accurate 44.1 kHz. */
  static const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = MP3_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = MP3_FRAME_SAMPLES,
    .use_apll = true,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
  };
#endif

  esp_err_t installStatus = i2s_driver_install(I2S_SPEAKER_PORT, &i2s_config, 0, NULL);
 if (ESP_OK != installStatus) {
    if (ESP_ERR_INVALID_ARG == installStatus) {
      Serial.println(" Error : i2s_driver_install BAD parameter");
       while(true);
    }
    if (ESP_ERR_NO_MEM == installStatus) {
      Serial.println(" Error : i2s_driver_install  out of memory");
       while(true);
    }
    Serial.println(" Error : i2s_driver_install  ????");
    while(true);
  }

  static const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCLK_PIN,
    .ws_io_num = I2S_LRC_PIN,
    .data_out_num = I2S_DOUT_PIN,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  installStatus = i2s_set_pin(I2S_SPEAKER_PORT, &pin_config);
  if (ESP_OK != installStatus) {
    if (ESP_ERR_INVALID_ARG == installStatus) {
      Serial.println(" Error : i2s_set_pin BAD parameter");
       while(true);
    }
    if (ESP_FAIL == installStatus) {
      Serial.println(" Error : i2s_set_pin   I/O Error");
       while(true);
    }
    Serial.println(" Error : i2s_set_pin  ????");
    while(true);
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

  //TEST WITHOUT IT
  //stopMicForPlayback();

  setupSpeakerI2S();

  size_t sampleIndex = 0;
  while (sampleIndex < totalSamples) {
    size_t n = fillSineChunk(chunk, PLAY_CHUNK_SAMPLES, sampleIndex, totalSamples);
    playPcm16(chunk, n, errorBlinkLed);
    sampleIndex += n;
  }

  // Let DMA drain the last buffers before uninstall.
  delay(50);
  //TEST WITHOUT IT
  //teardownSpeakerI2S();
  //restoreMicI2S();
  Serial.println("PLAY_DONE");
}

#endif // _MY_DMA_I2S_PLAY_H
