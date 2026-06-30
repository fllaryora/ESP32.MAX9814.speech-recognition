#include <Arduino.h>
#include "driver/i2s.h"
#include "driver/adc.h"
#include "esp_heap_caps.h"

//=== AUDIO SETTINGS ===
#define LED_PIN 2  // D2 //NO PIN
#define SAMPLE_RATE 16000 //Hz - No enough memory for 44100hz
#define RECORDING_DURATION 3 //sec

#define I2S_PORT I2S_NUM_0 // GPIO36 // PIN VP
#define ADC_CHANNEL ADC1_CHANNEL_0 // GPIO36 // PIN VP

#define DO_NOT_FORSAKE_IN_CASE_OF_TIME_OUT portMAX_DELAY

#define SERIAL_BAUD 115200

#define TOTAL_SAMPLES ( (size_t) (SAMPLE_RATE * RECORDING_DURATION) )

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

void setupI2S() {

  static const i2s_config_t i2s_config = {
    // 12 bit of resolution
    //.mode = (i2s_mode_t) (I2S_MODE_MASTER | I2S_MODE_ADC_BUILT_IN),
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT, //waste 4bits per sample.
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,

    .communication_format = I2S_COMM_FORMAT_I2S_LSB,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
    .use_apll = true,//para un sample rate raro
    .tx_desc_auto_clear = false,//true 
    .fixed_mclk = 0
  };

  esp_err_t installStatus = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);

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

  //la configuracion de ADC es cuando se usa un microfono señal analogica
  // en caso de usar un microfono con I2S como el INMP441 se configura con i2s_set_pin

  esp_err_t modeStatus = i2s_set_adc_mode(ADC_UNIT_1, ADC_CHANNEL);
  if (ESP_OK != modeStatus) {
    if (ESP_ERR_INVALID_ARG == modeStatus) {
      Serial.println(" Error : i2s_set_adc_mode BAD parameter");
       while(true);
    }
    Serial.println(" Error : i2s_set_adc_mode  ????");
    while(true);
  }

  // max ESP 12 bits resolution
  esp_err_t configStatus = adc1_config_width(ADC_WIDTH_BIT_12);
  if (ESP_OK != configStatus) {
    if (ESP_ERR_INVALID_ARG == configStatus) {
      Serial.println(" Error : adc1_config_width BAD parameter");
       while(true);
    }
    Serial.println(" Error : adc1_config_width  ????");
    while(true);
  }
  //the read is a comparation against to known voltage
  //3.3v require -11 db
  esp_err_t attenStatus = adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN_DB_12);
   if (ESP_OK != attenStatus) {
    if (ESP_ERR_INVALID_ARG == attenStatus) {
      Serial.println(" Error : adc1_config_channel_atten BAD parameter");
       while(true);
    }
    Serial.println(" Error : adc1_config_channel_atten  ????");
    while(true);
  }
  esp_err_t portStatus = i2s_adc_enable(I2S_PORT);
  if (ESP_OK != portStatus) {
    if (ESP_ERR_INVALID_ARG == portStatus) {
      Serial.println(" Error : i2s_adc_enable BAD parameter");
       while(true);
    }
    if (ESP_ERR_INVALID_STATE == portStatus) {
      Serial.println(" Error : i2s_adc_enable invalid state");
       while(true);
    }
    Serial.println(" Error : i2s_adc_enable  ????");
    while(true);
  }

}

void setup() {
  //speed of comunication with computer:
  Serial.begin(SERIAL_BAUD);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  malloc_audio_buffer();

  setupI2S();

 
  //Serial.println("Setup complete");
 

  digitalWrite(LED_PIN, HIGH);

}

void recordAudio() {
   //Serial.println("The fucking recording has started...");
   size_t totalRead = 0;
   //i2s use a pointer of 8
   uint8_t * headPointer = (uint8_t *) rawAudioData; // Audio buffer
   do {
     uint8_t * startPointer = headPointer + totalRead;
     size_t amountToRead = totalSamplesInBytes- totalRead;
      // i2s_read will return the amount of bytes that will have read into bytesbytesRead
     size_t bytesRead = 0;
     esp_err_t statusRead = i2s_read(I2S_PORT,
       startPointer,
       amountToRead,
       &bytesRead,
       DO_NOT_FORSAKE_IN_CASE_OF_TIME_OUT );
    if(ESP_OK != statusRead) {
      //leer el puto codigo
      //Serial.println(esp_err_to_name(statusRead));
      while(true){
        digitalWrite(LED_PIN, LOW);
        delay(500);
        digitalWrite(LED_PIN, HIGH);
      }
    }
     totalRead += bytesRead;
     //Serial.println("The fucking recording has finished...");

   } while(totalRead < totalSamplesInBytes);

   
   for(size_t time = 0; time < TOTAL_SAMPLES; time++){
     uint16_t raw = rawAudioData [time] & 0x0FFF;
     //centered of the wave in 0.
     rawAudioData [time] = (uint16_t) raw - 2048;

   }

}

void sendAudioSerial() {
  Serial.write("START", 5);
  Serial.write( (uint8_t*) &serialInBytes, sizeof(serialInBytes));
  Serial.write( (uint8_t*) rawAudioData, serialInBytes);
  Serial.write("END", 3);
}

void loop() {
 if(Serial.available()) {
   char order = Serial.read();
   bool isRecordOrder = order == 'r';
   if(isRecordOrder) {
    recordAudio();
    sendAudioSerial();
   }

 }

}
