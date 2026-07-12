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
  //uint8_t* probabilities;
  int8_t* probabilities;
   //TfLiteTensor* output = interpreter->output(0);
  // Array for storing the model’s input, output and intermediate tensors.
  uint8_t *tensor_arena; //100 *1024
  tflite::ErrorReporter* error_reporter;
};



void setupCNN(ConvNeurNetwork* cnn, uint8_t errorBlinkLed) {
  //static variables are not deleted after go out from the function
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

  //static variables are not deleted after go out from the function
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

  //static variables are not deleted after go out from the function
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

  //cnn->probabilities = cnn->interpreter->output(0)->data.uint8;
  //the real type is in the model read it with netron or directly fron the ESP32
  //if you get the wrong pointer wrong pregiction you will get
  cnn->probabilities = cnn->interpreter->output(0)->data.int8;

}


/************************************************
* The function returns the name of the category detected by the sensor.
*
* int kCategoryCount  	- The number of classes that the model can predict.
* uint8_t probabilities[]  - An array to store the probabilities for all classes.
* char* kCategoryLabels[] - An array of the names of the categories that the model can classify.
************************************************/
//String getPrediction(int kCategoryCount, uint8_t probabilities [], const char* kCategoryLabels[]){
String getPrediction(int kCategoryCount, int8_t probabilities [], const char* kCategoryLabels[]){
 
  int maxProbabilityIndex = 0;
  int8_t maxProbability = probabilities[maxProbabilityIndex];

  for (int index = 1; index < kCategoryCount; index++) {
    if(probabilities[index] > maxProbability){
      maxProbability = probabilities[index];
      maxProbabilityIndex = index;
    }
  }

  return String(kCategoryLabels[maxProbabilityIndex]);
}


struct PredictionResult
{
	int index;
	int8_t raw;
	float probability;
};

void getProbabilitiesOrdered(
	int categoryCount,
	int8_t probabilities[],
	float outputScale,
	int outputZeroPoint,
	PredictionResult results[])
{
	// Copy and decuantize
	for(int i = 0; i < categoryCount; i++)
	{
    	results[i].index = i;
    	results[i].raw = probabilities[i];

    	results[i].probability = ((float)probabilities[i] - outputZeroPoint) * outputScale;
	}

	// Desc order
	for(int i = 0; i < categoryCount - 1; i++)
	{
    	for(int j = i + 1; j < categoryCount; j++)
    	{
        	if(results[j].probability > results[i].probability)
        	{
            	PredictionResult tmp = results[i];
            	results[i] = results[j];
            	results[j] = tmp;
        	}
    	}
	}
}
#endif  // TENSORFLOW_LITE_CONFIG

