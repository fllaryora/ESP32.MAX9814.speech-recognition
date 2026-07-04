#include <Arduino.h>

#include "./dma_malloc.h"
#include "./dma_i2s_record.h"
#include "./pcm_to_spectrogram.h"

#define LED_PIN 2  // BLUE LED on ESP32 - D2 //NO PIN
#define SERIAL_BAUD 115200
// other speeds
//230_400
//460_800
//500_000
//921_600

uint16_t * rawAudioData = nullptr; // Audio buffer
float*  spectrogramOutput = nullptr; // image buffer
Spectrogram* memory  = nullptr; //  configuration structure

static const uint32_t serialInBytes = TOTAL_SAMPLES_IN_BYTES;
static const uint32_t serialFramesInBytes = SPECTRUM_OUTPUT_SIZE * sizeof(float);

void setup() {
  //speed of comunication with computer:
  Serial.begin(SERIAL_BAUD);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  uint8_t errorBlinkLed = LED_PIN;

  rawAudioData = malloc_dma_buffer(TOTAL_SAMPLES_IN_BYTES, errorBlinkLed);
  spectrogramOutput = (float*) malloc_dma_buffer(SPECTRUM_OUTPUT_SIZE * sizeof(float), errorBlinkLed);
  memory = (Spectrogram*) malloc_dma_buffer(sizeof(Spectrogram), errorBlinkLed);
  memory->rawAudioDataInPcm = (int16_t * ) rawAudioData;
  memory->spectrogramOutput = spectrogramOutput;
  
  memory->kissFFTConfiguration = NULL;
  memory->smoothedNoiseFloor = 0.0f;
  initializeHanningWindows(memory);
  memory->kissFFTConfiguration = kiss_fftr_alloc(SAMPLES_FOR_EACH_FFT, 0, NULL, NULL);

  bool isFFTConfigurationNotInitialized = !(memory->kissFFTConfiguration);
  if (isFFTConfigurationNotInitialized) {
    Serial.println("ERROR: kiss_fftr_alloc() FAILED! Returned NULL!");
    Serial.println("Check: SAMPLES_FOR_EACH_FFT must be > 0");
    Serial.println("Check: Available RAM");
    while(true);
  }

  setupI2S();

  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);
  delay(500);
  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);

}

void sendAudioPCMSerial() {
  Serial.write("START", 5);
  Serial.write( (uint8_t*) &serialInBytes, sizeof(serialInBytes));
  Serial.write( (uint8_t*) rawAudioData, serialInBytes);
  Serial.write("END", 3);
}

void sendSpectrumSerial() {
  Serial.write("FRAME", 5);
  Serial.write( (uint8_t*) &serialFramesInBytes, sizeof(serialFramesInBytes));
  Serial.write( (uint8_t*) memory->spectrogramOutput, serialFramesInBytes);
  Serial.write("END", 3);
}

void loop() {
 if(Serial.available()) {
   char order = Serial.read();
   bool isRecordOrder = order == 'r';
   if(isRecordOrder) {
    
    uint8_t errorBlinkLed = LED_PIN;
    uint8_t recordingLed = LED_PIN;
    recordAudio(rawAudioData, recordingLed, errorBlinkLed);
    
    sendAudioPCMSerial();
    get_spectrogram( memory );
    sendSpectrumSerial();
   }

 }

}
