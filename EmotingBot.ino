/*Written by Julia Makivic, June 2020*/
/*Bits of this code were taken from the IMU_Classifier on the Arduino TensorflowLite github tutorial.*/

/*This code classifies gestures captured by the accelerometer and gyroscope on the Arduino Nano 33 BLE Sense.
The gestures are classified into two groups, a Wave or a High Five. Depending on the gesture, an LED Matrix
connected to the Arduino will display a distinct facial expression.*/

/*Include a library for the built in accelerometer and gyroscope*/
#include <Arduino_LSM9DS1.h>

/*Include files for TensorflowLite*/
#include <TensorFlowLite.h>
#include <tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h>
#include <tensorflow/lite/experimental/micro/micro_error_reporter.h>
#include <tensorflow/lite/experimental/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

/*Include the file containing the tensorflowlite model*/
#include "model.h"

/*Include files necessary for the LED Matrix*/
#include <Wire.h>
#include <Adafruit_GFX.h>
#include "Adafruit_LEDBackpack.h"

/*Parameters that determine the rate at which the accelerometer processes movement*/
const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 119;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::ops::micro::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize];

// array to map gesture index to a name
/*This array should include all of the gestures that we are trying to classify*/
const char* GESTURES[] = {
  "wave",
  "highfive" 
};

/*this constant defines the total number of gestures in the array */
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

//declaring the LED matrix
Adafruit_8x8matrix matrix = Adafruit_8x8matrix();

//Data structures to store the expressions of Emoting Bot
//These arrays determine which pixel on the 8x8 LED matrix will light up
//Think of each pixel as a byte that can be represented by 0 or 1. There are 8 bytes across and along the LED matrix. If the byte is 0, then that pixel is off, if the byte is 1, then the pixel is on.
//By combining 0s and 1s in each row, we can depict a facial expression on the LED matrix.

static const uint8_t PROGMEM
smile[] = {
  B00000000,
  B01100110,
  B01100110,
  B00000000,
  B01000010,
  B00100100,
  B00011000,
  B00000000
  },
winky_face[] ={
  B00000010,
  B00000100,
  B01101000,
  B00000000,
  B01000010,
  B01000010,
  B01111110,
  B00000000
  
  };

void setup() {
  /*Initializing serial communication*/
  Serial.begin(9600);
  while (!Serial);

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  //setting up LED Matrix
  Serial.println("8x8 LE Matrix Test");
  matrix.begin(0x70);
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // wait for significant motion
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      // read the acceleration data
      IMU.readAcceleration(aX, aY, aZ);

      // sum up the absolutes
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // check if it's above the threshold
      if (aSum >= accelerationThreshold) {
        // reset the sample read count
        samplesRead = 0;
        break;
      }
    }
  }

  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples) {
    // check if new acceleration AND gyroscope data is available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // normalize the IMU data between 0 to 1 and store in the model's
      // input tensor
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples) {
        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
          return;
        }

        // Loop through the output tensor values from the model
        for (int i = 0; i < NUM_GESTURES; i++) {
          Serial.print(GESTURES[i]);
          Serial.print(": ");
          //this conditional statement checks to see if the index in the GESTURES array == wave, and if tensorflow has determined that the gesture captured by the accelerometer closely matches the wave gesture
          //if this condition is true, then draw a smile on the LED matrix
          if(GESTURES[i] == "wave" && tflOutputTensor->data.f[i]>0.6){
            matrix.clear();
            matrix.drawBitmap(0,0, smile, 8,8, LED_ON);
            matrix.writeDisplay();
            }
            //if the index in the GESTURE array == highfive, and if tensorflow has determine that the gestur closely matches a highfive,
            //then draw a winky face on the LED matrix
            else if(GESTURES[i] == "highfive" && tflOutputTensor->data.f[i]>0.6 ){
              matrix.clear();
              matrix.drawBitmap(0,0, winky_face, 8,8, LED_ON);
              matrix.writeDisplay();
              }
          Serial.println(tflOutputTensor->data.f[i], 6);

          //this is where you show the changes on the LED Matrix
        }
        Serial.println();
      }
    }
  }
}
