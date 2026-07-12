#include <Arduino.h>

#include "./dma_malloc.h"
#include "./dma_i2s_record.h"
#include "./pcm_to_spectrogram.h"
#include "./TensorFlowLiteModelConfig.h"

#define LED_PIN 2  // BLUE LED on ESP32 - D2 //NO PIN
#define SERIAL_BAUD 460800 //460800//115200
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

/*
  const char* tensorTypeToString (TfLiteType type)
  {
    switch(type)
    {
        case kTfLiteNoType:   return "NoType";
        case kTfLiteFloat32:  return "Float32";
        case kTfLiteInt32:	return "Int32";
        case kTfLiteUInt8:	return "UInt8";
        case kTfLiteInt64:	return "Int64";
        case kTfLiteString:   return "String";
        case kTfLiteBool: 	return "Bool";
        case kTfLiteInt16:	return "Int16";
        case kTfLiteComplex64:return "Complex64";
        case kTfLiteInt8: 	return "Int8";
        default:          	return "Unknown";
    }
  };
*/
void sendSpectrumSerial() {
  Serial.write("FRAME", 5);
  Serial.write( (uint8_t*) &serialFramesInBytes, sizeof(serialFramesInBytes));
  Serial.write( (uint8_t*) memory->spectrogramOutput, serialFramesInBytes);
  Serial.write("END", 3);
}

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
  
  /**********DEPURATION VALUES************/
  /*
  TfLiteTensor* input  = cnn->interpreter->input(0);
  TfLiteTensor* output = cnn->interpreter->output(0);

  Serial.println();
  Serial.println("========== INPUT(0) ==========");

  Serial.print("Type   	: ");
  Serial.println((char *) tensorTypeToString(input->type));

  Serial.print("Bytes  	: ");
  Serial.println(input->bytes);

  Serial.print("Scale  	: ");
  Serial.println(input->params.scale, 8);

  Serial.print("Zero Point : ");
  Serial.println(input->params.zero_point);

  Serial.print("Shape  	: [");

  for(int i = 0; i < input->dims->size; i++)
  {
    Serial.print(input->dims->data[i]);

    if(i < input->dims->size - 1)
    {
        Serial.print(", ");
    }
  }

  Serial.println("]");


  Serial.println();
  Serial.println("========== OUTPUT(0) ==========");

  Serial.print("Type   	: ");
  Serial.println((char *) tensorTypeToString(output->type));

  Serial.print("Bytes  	: ");
  Serial.println(output->bytes);

  Serial.print("Scale  	: ");
  Serial.println(output->params.scale, 8);

  Serial.print("Zero Point : ");
  Serial.println(output->params.zero_point);

  Serial.print("Shape  	: [");

  for(int i = 0; i < output->dims->size; i++)
  {
    Serial.print(output->dims->data[i]);

    if(i < output->dims->size - 1)
    {
        Serial.print(", ");
    }
  }

  Serial.println("]");

  Serial.println("===============================");
  */
  /**********END DEPURATION VALUES************/

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

void loop() {
  uint8_t errorBlinkLed = LED_PIN;
  uint8_t recordingLed = LED_PIN;

  recordAudio(rawAudioData, recordingLed, errorBlinkLed);
  get_spectrogram( memory );
  //Serial.printf("Noise: %f\n", memory->smoothedNoiseFloor);
  //if(memory->smoothedNoiseFloor > 2.2f) {
    //sendSpectrumSerial()
    runCNN( cnn, errorBlinkLed);
    //for( int classId = 0; classId < kCategoryCount; classId++ ){
    //  Serial.printf("Prob %s = %d \n", kCategoryLabels[classId], cnn->probabilities[classId]);
    //}

    //String prediction = getPrediction( kCategoryCount, cnn->probabilities, kCategoryLabels);
    //Serial.printf("Prediction: %s\n", prediction);
    PredictionResult results[kCategoryCount];

    TfLiteTensor* output =
      cnn->interpreter->output(0);

    getProbabilitiesOrdered(
      kCategoryCount,
      cnn->probabilities,
      output->params.scale,
      output->params.zero_point,
      results
    );

    Serial.println("===== Ranking =====");

    for(int i = 0; i < kCategoryCount; i++)
    {
      Serial.printf(
          "%d) %s | raw=%d | prob=%f\n",
          i + 1,
          kCategoryLabels[results[i].index],
          results[i].raw,
          results[i].probability
      );
    }

    Serial.println("===================");


 
  //}
  
}
