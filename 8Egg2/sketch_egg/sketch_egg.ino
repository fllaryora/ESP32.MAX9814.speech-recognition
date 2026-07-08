#include <Arduino.h>

#include "./dma_malloc.h"
#include "./dma_i2s_record.h"
#include "./pcm_to_spectrogram.h"
#include "./TensorFlowLiteModelConfig.h"

#define LED_PIN 2  // BLUE LED on ESP32 - D2 //NO PIN
#define SERIAL_BAUD 115200 //460800//115200
// other speeds
//230_400
//460_800
//500_000
//921_600

uint16_t * rawAudioData = nullptr; // Audio buffer
float*  spectrogramOutput = nullptr; // image buffer
Spectrogram* memory  = nullptr; //  configuration structure
static uint8_t *tensor_arena = nullptr;
ConvNeurNetwork* cnn = nullptr;

static const uint32_t serialInBytes = TOTAL_SAMPLES_IN_BYTES;
static const uint32_t serialFramesInBytes = SPECTRUM_OUTPUT_SIZE * sizeof(float);
constexpr int kCategoryCount = 6;
const char* kCategoryLabels[kCategoryCount] = {"0_Ayuda", "1_Basura", "2_Listo", "3_No", "4_Papel", "5_Si"};

void setup() {
  //speed of comunication with computer:
  Serial.begin(SERIAL_BAUD);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  uint8_t errorBlinkLed = LED_PIN;
  //=== First stage == initialization
  rawAudioData = malloc_dma_buffer(TOTAL_SAMPLES_IN_BYTES, errorBlinkLed);
  
  tensor_arena = (uint8_t *) malloc_dma_buffer(TENSOR_ARENA_SIZE , errorBlinkLed);
  cnn = (ConvNeurNetwork*) malloc_dma_buffer(sizeof(ConvNeurNetwork), errorBlinkLed);
  cnn->tensor_arena = tensor_arena;
  setupCNN(cnn, errorBlinkLed);

  //To write the spectrogram strightfoward on CNN input
  spectrogramOutput =  cnn->input_data;
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

void loop() {
  uint8_t errorBlinkLed = LED_PIN;
  uint8_t recordingLed = LED_PIN;
  recordAudio(rawAudioData, recordingLed, errorBlinkLed);
  get_spectrogram( memory );

  if(memory->smoothedNoiseFloor > 2.2f ){
    runCNN( cnn, errorBlinkLed);
    String prediction = getPrediction( kCategoryCount, cnn->probabilities, kCategoryLabels);
    Serial.printf("Prediction: %s\n", prediction);
  }

}
