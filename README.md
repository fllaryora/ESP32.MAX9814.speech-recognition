# ESP32.MAX9814.speech-recognition

## Special thanks to Deniss Borisovich Stepanyuk REPO: https://github.com/DenissStepanjuk/ESP32.INMP441.speech-recognition/tree/main

## 1sketch_volt Folder

1sketch_volt Use it to read the output of voltaje of potenciometer of 10k ohms and see if the port VP work. If it output 0V, use a multimeter to read what the potenciometer get.

## 2sketch_jun27a Folder
2sketch_jun27a Will output the avoilable memory of your ESP32, I used it in order to determine why my mallocs were failing.

## 3greedy Folder
3greedy will test the allocation of big chunks of memory DMA that you will need. 

## 4Egg1 Folder
4Egg1:
 get the sound from max9814,
 get the spectrum and send both to the pc.
 Using python it re-generate the process of ESP 32 in order to later: create a model and learn. 
 
## DATASET GATHERIN
Use the code of 4Egg1,
but In the esp32, do not set the spectrum
speedup the baudrate in both places.
but In the PC, do not ask for the spectrum, and do not compare. just save the wavs

## DATASET DATA AUGmentation
Use 5Augmentation to create wavs with noise.
