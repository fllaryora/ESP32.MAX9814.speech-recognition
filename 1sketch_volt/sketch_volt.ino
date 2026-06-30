#include <Arduino.h>
#include "driver/i2s.h"
#include "driver/adc.h"
#include "esp_heap_caps.h"

//=== AUDIO SETTINGS ===
#define LED_PIN 2  // D2
#define SAMPLE_RATE 16000 //Hz - No enough memory for 44100hz
#define RECORDING_DURATION 3 //sec

#define I2S_PORT I2S_NUM_0 // GPIO36
#define ADC_CHANNEL ADC1_CHANNEL_0 // GPIO36
#define DO_NOT_FORSAKE_IN_CASE_OF_TIME_OUT portMAX_DELAY

#define SERIAL_BAUD 115200

#define TOTAL_SAMPLES ( (size_t) (SAMPLE_RATE * RECORDING_DURATION) )

//There are 327680bytes in the fucking heap
uint16_t * rawAudioData = nullptr; // Audio buffer
static const size_t totalSamplesInBytes = TOTAL_SAMPLES * sizeof(uint16_t);
static const uint32_t serialInBytes = TOTAL_SAMPLES * sizeof(uint16_t);


bool malloc_audio_buffer() {

  size_t largestBlock = heap_caps_get_largest_free_block(MALLOC_CAP_DMA);
  if(largestBlock < totalSamplesInBytes) {
    Serial.printf(" Error : Not enough memory . There is %u bytes free. But  %u were requested \n\n",largestBlock, totalSamplesInBytes);
    while(true);
  }
  rawAudioData = (uint16_t *) heap_caps_malloc(totalSamplesInBytes, MALLOC_CAP_DMA);
  if (!rawAudioData) {
    Serial.println(" Error : Failed to allocate memory");
    while(true);
  }

  return true;
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  

  //speed of comunication with computer:
  Serial.begin(SERIAL_BAUD);

  malloc_audio_buffer();

  //setupI2S();

  digitalWrite(LED_PIN, HIGH);
  Serial.println("Setup complete");

 
  //recordAudioI2S();
  //sendVoltage();

}

const int analogPin = 36; // Pin connected to your analog signal
float voltage = 0.0;      // Variable to store converted float

void loop() {
  // 1. Read the 12-bit raw ADC value (range from 0 to 4095)
  int rawValue = analogRead(analogPin);
  
  // 2. Convert to voltage (assuming 3.3V max and 12-bit resolution)
  voltage = (rawValue * 3.3) / 4095.0;

  // 3. Send the float value to the Serial Monitor
  Serial.print("Raw Value: ");
  Serial.print(rawValue);
  Serial.print(" | Calculated Voltage: ");
  Serial.print(voltage, 2); // Print with 2 decimal places
  Serial.println(" V");

  delay(250); // Wait half a second
}
