#ifndef TENSORFLOW_LITE_CONFIG
#define TENSORFLOW_LITE_CONFIG

// GET spectrogram constants
#include "./pcm_to_spectrogram.h"
#include "./TensorFlowLiteModel.h"
// TensorFlowLite_ESP32-
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"


// amount of rows (vector / tensor) send to the model (41x299 = 12 259)
constexpr int height = AMOUNT_OF_FRAMES_IN_OUTPUT; //299

// Elements in each row
constexpr int wide = POOLED_BINS_LENGTH; //41

// All values
constexpr int inputVectoSize = SPECTRUM_OUTPUT_SIZE; //12 259

//Amount to classes to predict
constexpr int kCategoryCount = 6;

// Names of the categories that the model can predict
const char* kCategoryLabels[kCategoryCount] = {"0_Ayuda", "1_Basura", "2_Listo", "3_No", "4_Papel", "5_Si"};

// Initialise the data structures required to work with the TensorFlow Lite library.
// The ErrorReporter object provides error reports and debugging information.
tflite::ErrorReporter* error_reporter = nullptr;

// Object for storing the machine learning model responsible for classifying images.
const tflite::Model* model = nullptr;

// Object for storing the interpreter responsible for loading and executing the model.
tflite::MicroInterpreter* interpreter = nullptr;

// Object (tensor) for storing the image sent to the model’s input for subsequent classification.
TfLiteTensor* input = nullptr;

// Amount of memory to be reserved for storing the model’s tensors.
// For the model’s input, output and intermediate tensors.
//constexpr int kTensorArenaSize = 81 * 1024;
constexpr int kTensorArenaSize = 100 * 1024;

// Array for storing the model’s input, output and intermediate tensors.
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external

/************************************************
* The function returns the name of the category detected by the sensor.
*
* int kCategoryCount  	- The number of classes that the model can predict.
* int8_t probabilities[]  - An array to store the probabilities for all classes.
* char* kCategoryLabels[] - An array of the names of the categories that the model can classify.
************************************************/
String getPrediction(int kCategoryCount, uint8_t probabilities [], const char* kCategoryLabels[]){
 
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

void setupCNN(uint8_t errorBlinkLed) {

  
  // To log errors, we create the variable ‘error_reporter’ using the structures provided by TensorFlow Lite.
  static tflite::MicroErrorReporter micro_error_reporter;

  // The ‘error_reporter’ variable must be passed to the interpreter, which will in turn send the list of errors to it.
  error_reporter = &micro_error_reporter;

  // Create an instance of the model using the data array defined in ‘TensorFlowLiteModel.h’
  model = tflite::GetModel(model_TFLite);

  // Check that the model version is compatible with the library version.
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                        "Model provided is schema version %d not equal "
                        "to supported version %d.",
                        model->version(), TFLITE_SCHEMA_VERSION);
    while(true){
          digitalWrite(errorBlinkLed, LOW);
          delay(500);
          digitalWrite(errorBlinkLed, HIGH);
          delay(500);
    }
  }

  // Allocate memory for the model’s input, output and intermediate tensors.
  if (tensor_arena == NULL) {
    // Allocate slower but higher-capacity memory.
    //tensor_arena = (uint8_t*) ps_calloc(kTensorArenaSize, 1);

    // Allocate faster memory with a lower capacity.
	  //tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

	  tensor_arena = (uint8_t*) malloc(kTensorArenaSize);
  }

  // If it was not possible to allocate memory for the model’s tensors, display an error.
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    while(true){
            digitalWrite(errorBlinkLed, LOW);
            delay(500);
            digitalWrite(errorBlinkLed, HIGH);
            delay(500);
      }
  }


  // Load all the methods included in TensorFlow Lite to process data using the model.
  // (It consumes a large amount of memory)
  // tflite::AllOpsResolver resolver;

  // Load only the necessary methods from TensorFlow Lite.
  static tflite::MicroMutableOpResolver<9> micro_op_resolver;
  // WTF i that 9? 9 is not a generic type like java
  // Non-type template parameter, it is going to have 9 operators (neuron layers)
  //at compiler time it store 9 places
  //1
  micro_op_resolver.AddAveragePool2D();
  // 2
  micro_op_resolver.AddMaxPool2D();
  // 3 Reshape: an operation used in machine learning and data processing
  // that changes the shape (dimensionality) of a tensor without altering its data.
  micro_op_resolver.AddReshape();
  // 4
  micro_op_resolver.AddFullyConnected();
  // 5 
  micro_op_resolver.AddConv2D();
  // 6 DepthwiseConv2D: a variant of a convolutional layer used to increase
  // computational efficiency and reduce the number of model parameters.
  micro_op_resolver.AddDepthwiseConv2D();
  // 7
  micro_op_resolver.AddSoftmax();
  // 8 Quantisation: the process of converting data or models to reduce
  // their size and computational complexity whilst maintaining an acceptable level of accuracy.
  micro_op_resolver.AddQuantize();
  //9 Dequantisation: the inverse process that converts quantised data
  // back to floating-point format or to a higher precision.
  micro_op_resolver.AddDequantize();

  // Create an instance of the interpreter, passing the data required to run the model.
  static tflite::MicroInterpreter static_interpreter(
	model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  	//model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);

  interpreter = &static_interpreter;


  // Allocate memory for the model’s internal tensors using the tensor_arena memory.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  // If the memory allocation fails, report the error.
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed. The sandpit ran out of sand.");
    while(true){
              digitalWrite(errorBlinkLed, LOW);
              delay(500);
              digitalWrite(errorBlinkLed, HIGH);
              delay(500);
    }
  }

  // Get a pointer to the model’s input tensor.
  input = interpreter->input(0);

}

String getClassFromSpectrogtam(Spectrogram * memory){
  // Model input tensor
	float * input_data = input->data.f;



	// It Transsfer the spectrogram (format NHWC: [frames][bins][1])
	// to the input of the CNN
	for (int pixelIndex = 0; pixelIndex < SPECTRUM_OUTPUT_SIZE; pixelIndex++) {
    input_data[pixelIndex] = memory->spectrogramOutput [pixelIndex];
	}

	// Run the model (convert the input image into probabilities
  // of belonging to each of the possible classes).
	if (kTfLiteOk != interpreter->Invoke()) {
  	TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
	}

	TfLiteTensor* output = interpreter->output(0);

	//int8_t probabilities [kCategoryCount];	
	//for(int classIndex = 0; classIndex < kCategoryCount; classIndex++){
  	//probabilities [classIndex] = output->data.uint8[classIndex];
	//}

  String prediction = getPrediction(kCategoryCount, output->data.uint8, kCategoryLabels);
  return prediction;
}

#endif  // TENSORFLOW_LITE_CONFIG

