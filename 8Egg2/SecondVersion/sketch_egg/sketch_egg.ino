#include <Arduino.h>
// arduino-cli compile --fqbn esp32:esp32:esp32 "Path"
#include "./dma_malloc.h"
#include "./dma_i2s_record.h"
#include "./pcm_to_spectrogram.h"
#include "./mel_helpers.h"
#include "./spectrogram_to_spot_word.h"

#define LED_PIN 2  // BLUE LED on ESP32 - D2 //NO PIN
#define SERIAL_BAUD 115200 //460800//115200
// other speeds
//230_400
//460_800
//500_000
//921_600

uint16_t * rawAudioData = nullptr; // Audio buffer
float*  melSpectrogramOutput = nullptr; // image buffer
Spectrogram* memory  = nullptr; //  configuration structure
static uint8_t *tensor_arena = nullptr;
ConvNeurNetwork* cnn = nullptr;

static const uint32_t serialInBytes = TOTAL_SAMPLES_IN_BYTES;
static const uint32_t serialFramesInBytes = SPECTRUM_OUTPUT_SIZE * sizeof(float);

void sendSpectrumSerial() {
  Serial.write("FRAME", 5);
  Serial.write( (uint8_t*) &serialFramesInBytes, sizeof(serialFramesInBytes));
  Serial.write( (uint8_t*) memory->melSpectrogramOutput, serialFramesInBytes);
  Serial.write("END", 3);
}

void sendAudioPCMSerial() {
  Serial.write("START", 5);
  Serial.write( (uint8_t*) &serialInBytes, sizeof(serialInBytes));
  Serial.write( (uint8_t*) rawAudioData, serialInBytes);
  Serial.write("END", 3);
}

void setup() {
  //speed of comunication with computer:
  Serial.begin(SERIAL_BAUD);
  //Warning some libs always send at 115200 bauds.
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  setupLED();

  uint8_t errorBlinkLed = LED_PIN;
  
  //=== First stage == initialization
  rawAudioData = malloc_dma_buffer(TOTAL_SAMPLES_IN_BYTES, errorBlinkLed);
  
  tensor_arena = (uint8_t *) malloc_dma_buffer(TENSOR_ARENA_SIZE , errorBlinkLed);
  
  cnn = (ConvNeurNetwork*) malloc_dma_buffer(sizeof(ConvNeurNetwork), errorBlinkLed);
  
  cnn->tensor_arena = tensor_arena;
  
  setupCNN(cnn, errorBlinkLed);
  
  //To write the spectrogram strightfoward on CNN input
  melSpectrogramOutput =  cnn->input_data;
  memory = (Spectrogram*) malloc_dma_buffer(sizeof(Spectrogram), errorBlinkLed);
  
  memory->rawAudioDataInPcm = (int16_t * ) rawAudioData;
  memory->melSpectrogramOutput = melSpectrogramOutput;
  
  memory->kissFFTConfiguration = NULL;
  initializeHanningWindows(memory);
  
  memory->kissFFTConfiguration = kiss_fftr_alloc(SAMPLES_FOR_EACH_FFT, 0, NULL, NULL);

  bool isFFTConfigurationNotInitialized = !(memory->kissFFTConfiguration);
  if (isFFTConfigurationNotInitialized) {
    Serial.println("ERROR: kiss_fftr_alloc() FAILED! Returned NULL!");
    Serial.println("Check: SAMPLES_FOR_EACH_FFT must be > 0");
    Serial.println("Check: Available RAM");
    while(true){
      digitalWrite(errorBlinkLed, HIGH);
      delay(500);
      digitalWrite(errorBlinkLed, LOW);
      delay(500);
    }
    
  }

  setupI2S();

  Serial.println("SETUP OK");
  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);
  delay(500);
  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);

}
/* USe this to make dataset and validate the spectrum that python is making
void setup() {
  //speed of comunication with computer:
  Serial.begin(SERIAL_BAUD);
  //Warning some libs always send at 115200 bauds.
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  uint8_t errorBlinkLed = LED_PIN;
  
  //=== First stage == initialization
  rawAudioData = malloc_dma_buffer(TOTAL_SAMPLES_IN_BYTES, errorBlinkLed);
  
  melSpectrogramOutput = (float*) malloc_dma_buffer(SPECTRUM_OUTPUT_SIZE * sizeof(float), errorBlinkLed);
  memory = (Spectrogram*) malloc_dma_buffer(sizeof(Spectrogram), errorBlinkLed);
  
  
  memory->rawAudioDataInPcm = (int16_t * ) rawAudioData;
  memory->melSpectrogramOutput = melSpectrogramOutput;
  
  memory->kissFFTConfiguration = NULL;
  initializeHanningWindows(memory);
  
  memory->kissFFTConfiguration = kiss_fftr_alloc(SAMPLES_FOR_EACH_FFT, 0, NULL, NULL);

  bool isFFTConfigurationNotInitialized = !(memory->kissFFTConfiguration);
  if (isFFTConfigurationNotInitialized) {
    Serial.println("ERROR: kiss_fftr_alloc() FAILED! Returned NULL!");
    Serial.println("Check: SAMPLES_FOR_EACH_FFT must be > 0");
    Serial.println("Check: Available RAM");
    while(true){
      digitalWrite(errorBlinkLed, HIGH);
      delay(500);
      digitalWrite(errorBlinkLed, LOW);
      delay(500);
    }
    
  }

  setupI2S();

  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);
  delay(500);
  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);

}*/

/*
void loop() {

  uint8_t errorBlinkLed = LED_PIN;
  uint8_t recordingLed = LED_PIN;

  if(Serial.available()) {
    char order = Serial.read();
    bool isRecordOrder = order == 'r';
    if(isRecordOrder) {

      recordAudio(rawAudioData, recordingLed, errorBlinkLed);
      bool voiceDetected = get_spectrogram( memory );
      sendAudioPCMSerial();
      sendSpectrumSerial();

    }

  }
}
*/

void loop() {
  uint8_t errorBlinkLed = LED_PIN;
  uint8_t recordingLed = LED_PIN;

  recordAudio(rawAudioData, recordingLed, errorBlinkLed);
   bool voiceDetected = get_spectrogram( memory );
  if(voiceDetected) {
    //Serial.printf("Energy: %f\n", memory->maxFrameEnergy);
    runCNN( cnn, errorBlinkLed);

    PredictionResult results[kCategoryCount];

    TfLiteTensor* output = cnn->interpreter->output(0);

    getProbabilitiesOrdered(
      kCategoryCount,
      cnn->probabilities,
      output->params.scale,
      output->params.zero_point,
      results
    );

    Serial.println("===== Ranking =====");

    for(int i = 0; i < 3; i++) {
      Serial.printf( "%d) %s | prob=%f\n", i + 1, kCategoryLabels[results[i].index], results[i].probability);
    }

    Serial.println("===================");
    setLedsByPrediction(results[0].index);
 
  }
  
}
