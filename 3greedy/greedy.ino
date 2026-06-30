#include <Arduino.h>
#include "esp_heap_caps.h"


size_t printHeapInfo(uint32_t caps, const char * name) {

  size_t largestBlock = heap_caps_get_largest_free_block(caps);
  


  Serial.printf(
    " %s \n\n", name
  );
  
  Serial.printf(
    "LARGEST BLOCK: %u bytes ===\n", largestBlock
  );

  return largestBlock;

}

void myMallocUpitero(size_t largestBlock) {
 void *  upiteriaONE = heap_caps_malloc(largestBlock, MALLOC_CAP_DMA);
  if(upiteriaONE == NULL) {
    Serial.printf(" FALLO %u",largestBlock );
  } else {
    Serial.printf(" EXITO bloque %u",largestBlock );
  }
}


void setup() {
  Serial.begin(115200);
  
  size_t largestBlock = printHeapInfo(MALLOC_CAP_DMA,"DMA ");
  myMallocUpitero(largestBlock);
  largestBlock = printHeapInfo(MALLOC_CAP_DMA,"DMA ");
  myMallocUpitero(largestBlock);
  largestBlock = printHeapInfo(MALLOC_CAP_DMA,"DMA ");
  myMallocUpitero(largestBlock);
  largestBlock = printHeapInfo(MALLOC_CAP_DMA,"DMA ");
  myMallocUpitero(largestBlock);
  
  Serial.println(" ======FIN de la trasmision ======");
  
}

void loop() {
  // put your main code here, to run repeatedly:

}
