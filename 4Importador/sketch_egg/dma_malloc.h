#ifndef _MY_DMA_MALLOC_H   
#define _MY_DMA_MALLOC_H
#include "esp_heap_caps.h"

uint16_t * malloc_dma_buffer(size_t lengthInBytes, uint8_t errorBlinkLed) {

  size_t largestBlock = heap_caps_get_largest_free_block(MALLOC_CAP_DMA);
  if(largestBlock < lengthInBytes) {
    Serial.printf(" Error : Not enough memory . There is %u bytes free. But  %u were requested \n\n",largestBlock, lengthInBytes);
     while(true){
        digitalWrite(errorBlinkLed, LOW);
        delay(500);
        digitalWrite(errorBlinkLed, HIGH);
        delay(500);
      }
  }
  uint16_t * rawBufferData = (uint16_t *) heap_caps_malloc(lengthInBytes, MALLOC_CAP_DMA);
  if (!rawBufferData) {
    Serial.println(" Error : Failed to allocate memory");
     while(true){
        digitalWrite(errorBlinkLed, LOW);
        delay(500);
        digitalWrite(errorBlinkLed, HIGH);
        delay(500);
      }
  }

  return rawBufferData;
}

#endif // _MY_DMA_MALLOC_H
