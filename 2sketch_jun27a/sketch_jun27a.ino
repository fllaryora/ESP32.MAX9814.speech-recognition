#include <Arduino.h>
#include "esp_heap_caps.h"

void printHeapInfo(uint32_t caps, const char * name) {
  //can be used to return the current free memory
  // for different memory capabilities.
  size_t freeHeap = heap_caps_get_free_size(caps);
  //can be used to return the largest free block in the heap,
  // which is also the largest single allocation currently possible.
  // Tracking this value and comparing it to
  // the total free heap allows you to detect heap fragmentation.
  size_t largestBlock = heap_caps_get_largest_free_block(caps);
  //Get the total minimum free memory of all regions with the given capabilities.
  size_t minBlock = heap_caps_get_minimum_free_size(caps);
  //Get the total free size of all the regions that have the given capabilities. 
  size_t allRegions = heap_caps_get_free_size(caps);
  


  Serial.printf(
    "=== %s ===\n", name
  );
  Serial.printf(
    "Free HEAP: %u bytes ===\n", freeHeap
  );
  Serial.printf(
    "LARGEST BLOCK: %u bytes ===\n", largestBlock
  );

  Serial.printf(
    "MIN BLOCK: %u bytes ===\n", minBlock
  );

Serial.printf(
    "Free HEAP 2: %u bytes ===\n", allRegions
  );
//prints a summary of the information returned by heap_caps_get_info() to stdout.
  heap_caps_print_heap_info(caps);
  heap_caps_dump(caps);

}

void setup() {
  Serial.begin(115200);
  Serial.println(" ESP 32 HEAP");
  printHeapInfo(MALLOC_CAP_8BIT, "Malloc 8bit Data RAM heapeable. CPU accesible por bus de datos");

  //DMA == direct memory access
  printHeapInfo(MALLOC_CAP_DMA,
   "Allocate memory which is suitable for use with hardware DMA engines (for example SPI and I2S). This capability flag excludes any external PSRAM");
  
  //MALLOC_CAP_EXEC --> ejecutables solo
  printHeapInfo(MALLOC_CAP_32BIT, " MIX de data ram y instruction ram . no almacena datos float");
  //
  printHeapInfo(MALLOC_CAP_INTERNAL, "Memoria flash que no desaparece al apagar la cucaracha");
  
  Serial.println(" ======FIN de la trasmision ======");
  
}

void loop() {
  // put your main code here, to run repeatedly:

}
