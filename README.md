# ESP32.MAX9814.speech-recognition

## Special thanks to Deniss Borisovich Stepanyuk REPO: https://github.com/DenissStepanjuk/ESP32.INMP441.speech-recognition/tree/main

## https://github.com/tensorflow/tflite-micro

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
 
MAX9814 -> ESP32
GND -> GND
VDD+GAIN connected together, attached to 3.3v
OUT -> GPIO36


## DATASET GATHERIN
Use the code of 4Egg1,
but In the esp32, do not set the spectrum
speedup the baudrate in both places.
but In the PC, do not ask for the spectrum, and do not compare. just save the wavs

## DATASET DATA AUGmentation
Use 5Augmentation to create wavs with noise.

## 7Machinne Learning with tensor flow lite Folder
From dataset wavs it create the model.cc file which is necesary to have it in the ESP32.
---
Run 
xxd -i speech_commands_299x41.tflite > model2.cc

in order to get the model
## 8Egg2 Does not work because the CNN model is too huge for the ram of ESP32.
## 7Machinne Second version
I had to reduce de model in order to ESP32 be able to allocate memory
## 8Egg2 second version
My model is wrong, lot of bad predictions. and I have to depurate.

rst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)
configsip: 0, SPIWP:0xee
clk_drv:0x00,q_drv:0x00,d_drv:0x00,cs0_drv:0x00,hd_drv:0x00,wp_drv:0x00
mode:DIO, clock div:1
load:0x3fff0030,len:4876
ho 0 tail 12 room 4
load:0x40078000,len:16560
load:0x40080400,len:3500
entry 0x400805b4

========== INPUT(0) ==========
Type   	: Float32
Bytes  	: 49036
Scale  	: 0.00000000
Zero Point : 0
Shape  	: [1, 299, 41, 1]

========== OUTPUT(0) ==========
Type   	: Int8
Bytes  	: 6
Scale  	: 0.00390625
Zero Point : -128
Shape  	: [1, 6]
===============================
SETUP OK

I was expecting unit8, and I get signed int8.
6) 5_Si | raw=-128 | prob=0.000000
===================
===== Ranking =====
1) 1_Basura | raw=119 | prob=0.964844
2) 5_Si | raw=-119 | prob=0.035156
3) 2_Listo | raw=-128 | prob=0.000000
4) 3_No | raw=-128 | prob=0.000000
5) 4_Papel | raw=-128 | prob=0.000000
6) 0_Ayuda | raw=-128 | prob=0.000000
===================
===== Ranking =====
1) 5_Si | raw=127 | prob=0.996094
2) 1_Basura | raw=-128 | prob=0.000000
3) 2_Listo | raw=-128 | prob=0.000000
4) 3_No | raw=-128 | prob=0.000000
5) 4_Papel | raw=-128 | prob=0.000000
6) 0_Ayuda | raw=-128 | prob=0.000000
===================
===== Ranking =====
1) 5_Si | raw=127 | prob=0.996094

4) 3_No | raw=-128 | prob=0.000000
5) 4_Papel | raw=-128 | prob=0.000000
6) 5_Si | raw=-128 | prob=0.000000
===================
===== Ranking =====
1) 2_Listo | raw=127 | prob=0.996094
2) 1_Basura | raw=-128 | prob=0.000000
3) 0_Ayuda | raw=-128 | prob=0.000000
4) 3_No | raw=-128 | prob=0.000000
5) 4_Papel | raw=-128 | prob=0.000000
6) 5_Si | raw=-128 | prob=0.000000
===================
===== Ranking =====
1) 1_Basura | raw=127 | prob=0.996094
2) 0_Ayuda | raw=-128 | prob=0.000000
3) 2_Listo | raw=-128 | prob=0.000000
4) 3_No | raw=-128 | prob=0.000000
5) 4_Papel | raw=-128 | prob=0.000000
6) 5_Si | raw=-128 | prob=0.000000
===================

There is a lot of overfiting to choose 1_Basura (noise, conversations  and so on).



SD card adapter -> ESP32
CS -> IO5
SCK -> IO18
MOSI -> IO23
MISO -> IO19
VCC -> 3.3v
GND -> GND

