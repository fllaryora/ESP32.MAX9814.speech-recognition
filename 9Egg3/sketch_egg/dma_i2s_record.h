#ifndef _MY_DMA_I2S_RECORD_H   
#define _MY_DMA_I2S_RECORD_H   

#include "driver/i2s.h"
#include "driver/adc.h"

#define SAMPLE_RATE 16000 //Hz - No enough memory for 44100hz
#define I2S_PORT I2S_NUM_0 // refers to the first physical I2S hardware controller inside the ESP32 chip.
#define ADC_CHANNEL ADC1_CHANNEL_0 // GPIO36 // PIN VP
#define DO_NOT_FORSAKE_IN_CASE_OF_TIME_OUT portMAX_DELAY


//#define RECORDING_DURATION 3 //sec - No enough memory for 10 seconds
//#define TOTAL_SAMPLES ( (size_t) (SAMPLE_RATE * RECORDING_DURATION) )
#define TOTAL_SAMPLES ( (size_t) (24000) ) //1.5 segundos
#define TOTAL_SAMPLES_IN_BYTES ( (size_t) (TOTAL_SAMPLES * sizeof(uint16_t)) )

/*In case of error it will halt and send the cause by serial*/
void setupI2S() {

  static const i2s_config_t i2s_config = {
    // 12 bit of resolution
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT, //waste 4bits per sample.
    // it only listens to the "left channel" , the pin VP
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    //it set the communication format of protocol.
    .communication_format = I2S_COMM_FORMAT_I2S_LSB,
    // interruption flags///< Accept a Level 1 interrupt vector (lowest priority)
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    // amount of buffers of DMA (circular buffer DMA). The highest, the flowest of transfer, but the hiest of use of DMA.
    .dma_buf_count = 8,
    //total len of dma , this value must be less than 4092
    .dma_buf_len = 1024,
    .use_apll = true,// true = there is no good divisor of low frecuency   
    .tx_desc_auto_clear = false,// with false the sound is better 
    .fixed_mclk = 0// It set use_apll==true, so it has to set the mclk, mclk == 0 means the lib will set it for us.
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

  //the follow configuration is for a ADC microphone MAX 9814 by sending analog values.
  // In case of I2S microphone like INMP441 cpnfig with i2s_set_pin
//Built-in DAC functions are only supported on I2S0 for current ESP32 chip. 

//SET pin both channels: https://docs.espressif.com/projects/esp-idf/en/v4.2.3/esp32/api-reference/peripherals/i2s.html
/*Each controller can operate in half-duplex communication mode. Thus, the two controllers can be combined to establish full-duplex communication.*/
  esp_err_t modeStatus = i2s_set_adc_mode(ADC_UNIT_1, ADC_CHANNEL);
  if (ESP_OK != modeStatus) {
    if (ESP_ERR_INVALID_ARG == modeStatus) {
      Serial.println(" Error : i2s_set_adc_mode BAD parameter");
       while(true);
    }
    Serial.println(" Error : i2s_set_adc_mode  ????");
    while(true);
  }

  // max ESP 12 bits pin analog to digital converter resolution
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

  // the follow call will took all the  last configurations and analyze them all togather
  // In case of error it will send the specific cause in the serial print.
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

/* Tear down ADC mic I2S so speaker TX can run (half-duplex). */
void teardownI2S() {
  i2s_adc_disable(I2S_PORT);
  i2s_driver_uninstall(I2S_PORT);
}

/* record audio for 16 bit recording*/
/* rawBufferData is DMA memory*/
/* ledpPin is  output led initialized*/
/* In case of fail it will halt and ledpPin will blink*/
void recordAudio(uint16_t * rawBufferData, uint8_t recordingLed, uint8_t errorBlinkLed) {
  digitalWrite(recordingLed, HIGH);
   size_t totalRead = 0;
   uint8_t * headPointer = (uint8_t *) rawBufferData; // Audio buffer
   do {
     uint8_t * startPointer = headPointer + totalRead;
     size_t amountToRead = TOTAL_SAMPLES_IN_BYTES- totalRead;
      // i2s_read will return the amount of bytes that will have read into bytesbytesRead
     size_t bytesRead = 0;
     esp_err_t statusRead = i2s_read(I2S_PORT,
       startPointer,
       amountToRead,
       &bytesRead,
       DO_NOT_FORSAKE_IN_CASE_OF_TIME_OUT );
    if(ESP_OK != statusRead) {
      // In case you want to read the cause
      //Serial.println(esp_err_to_name(statusRead));
      digitalWrite(recordingLed, LOW);
      while(true){
        digitalWrite(errorBlinkLed, LOW);
        delay(500);
        digitalWrite(errorBlinkLed, HIGH);
        delay(500);
      }
    }
     totalRead += bytesRead;

   } while(totalRead < TOTAL_SAMPLES_IN_BYTES);

  digitalWrite(recordingLed, LOW);
  for(size_t time = 0; time < TOTAL_SAMPLES; time++){
     uint16_t raw = rawBufferData [time] & 0x0FFF;
     //centered of the wave in 0.
     rawBufferData [time] = (uint16_t) raw - 2048;
  }

  /**********DEPURATION VALUES************/
  /*
  uint16_t minSample = rawBufferData [0];
  uint16_t maxSample = rawBufferData [0];
  uint16_t meanSample = rawBufferData [0];

   for(size_t time = 1; time < TOTAL_SAMPLES; time++){
     if(rawBufferData [time] > maxSample) {
      maxSample = rawBufferData [time];
     }
     if(rawBufferData [time] < minSample) {
      minSample = rawBufferData [time];
     }
     meanSample += rawBufferData [time];

  }

  Serial.printf("PCM min=%d max=%d mean=%f", minSample, maxSample, meanSample/48000.0f);
  */
  /**********END DEPURATION VALUES************/

}

#endif // _MY_DMA_I2S_RECORD_H
