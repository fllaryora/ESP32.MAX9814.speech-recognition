#ifndef TENSORFLOW_LITE_CONFIG
#define TENSORFLOW_LITE_CONFIG

// The model and the neurons learnt
#include "./TensorFlowLiteModel.h"
// TensorFlowLite_ESP32-
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"

#define TENSOR_ARENA_SIZE ( (size_t) 100*1024)

struct ConvNeurNetwork {
  const tflite::Model* model;
  tflite::MicroInterpreter* interpreter;
  float * input_data;
  uint8_t* probabilities;
   //TfLiteTensor* output = interpreter->output(0);
  // Array for storing the model’s input, output and intermediate tensors.
  static uint8_t *tensor_arena; //100 *1024
  tflite::ErrorReporter* error_reporter;
};



void setupCNN(ConvNeurNetwork* cnn, uint8_t errorBlinkLed) {
 
  static tflite::MicroErrorReporter micro_error_reporter;

  cnn->model = tflite::GetModel(model_TFLite);

  // Check that the model version is compatible with the library version.
  if (cnn->model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter,
                        "Model provided is schema version %d not equal "
                        "to supported version %d.",
                        cnn->model->version(), TFLITE_SCHEMA_VERSION);
    while(true){
          digitalWrite(errorBlinkLed, LOW);
          delay(500);
          digitalWrite(errorBlinkLed, HIGH);
          delay(500);
    }
  }

  //store 9 operations at compile time
  static tflite::MicroMutableOpResolver<9> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(	cnn->model, micro_op_resolver, cnn->tensor_arena, TENSOR_ARENA_SIZE);

  cnn->interpreter = &static_interpreter;

  TfLiteStatus allocate_status = cnn->interpreter->AllocateTensors();

  // If the memory allocation fails, report the error.
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter, "AllocateTensors() failed. The sandpit ran out of sand.");
    while(true){
              digitalWrite(errorBlinkLed, LOW);
              delay(500);
              digitalWrite(errorBlinkLed, HIGH);
              delay(500);
    }
  }

  cnn->error_reporter = &micro_error_reporter;
  // Model input tensor
  //// input_data format NHWC (Batch or samples /height / Width / channels)
  //(format NHWC: [1 sample][299 frames][41 bins][1 channel]
  cnn->input_data = cnn->interpreter->input(0)->data.f;

}

void runCNN(ConvNeurNetwork* cnn, uint8_t errorBlinkLed){

	if (kTfLiteOk != cnn->interpreter->Invoke()) {
  	TF_LITE_REPORT_ERROR(cnn->error_reporter, "Invoke failed.");
    digitalWrite(errorBlinkLed, LOW);
              delay(500);
              digitalWrite(errorBlinkLed, HIGH);
              delay(500);
	}

  cnn->probabilities = cnn->interpreter->output(0)->data.uint8;

}


/************************************************
* The function returns the name of the category detected by the sensor.
*
* int kCategoryCount  	- The number of classes that the model can predict.
* uint8_t probabilities[]  - An array to store the probabilities for all classes.
* char* kCategoryLabels[] - An array of the names of the categories that the model can classify.
************************************************/
String getPrediction(int kCategoryCount, uint8_t probabilities [], const char* kCategoryLabels[]){
 
  int maxProbabilityIndex = 0;
  uint8_t maxProbability = probabilities[maxProbabilityIndex];

  for (int index = 1; index < kCategoryCount; index++) {
    if(probabilities[index] > maxProbability){
      maxProbability = probabilities[index];
      maxProbabilityIndex = index;
    }
  }

  return String(kCategoryLabels[maxProbabilityIndex]);
}

#endif  // TENSORFLOW_LITE_CONFIG

