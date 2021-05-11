/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* Notice: Marc Zalik has modified this file from the original code
==============================================================================*/

#include <TensorFlowLite.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "main_functions.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Global Variables
extern int pong_command;

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;

// Pong variables
int buttonState = 0;
int gameState = 0;                 // 0 = Home, 1 = Game, 2 = End

int controllerValue1 = 0;          // variable to store the value coming from the potentiometer
int controllerValue2 = 0;          // variable to store the value coming from the potentiometer

int paddlePositionPlayer1 = 0;
int paddlePositionPlayer2 = 0;  

int scorePlayer1 = 0;    
int scorePlayer2 = 0;

int ballX = 128/2;      
int ballY = 64/2;
int ballSpeedX = 2;
int ballSpeedY = 1;


// Display variables
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

#define OLED_RESET 4
#define SCREEN_ADDRESS 0x3D
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET); 
    
#if (SSD1306_LCDHEIGHT != 64)
  #error("Height incorrect, please fix Adafruit_SSD1306.h!");
#endif
}  // namespace

// Initializations
void setup() {
  // Pong setup
  Serial.begin(9600);
  display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS);  // initialize with the I2C addr 0x3C (for the 128x64)
  display.clearDisplay();

  // Starting paddle locations
  controllerValue1 = 500;
  controllerValue2 = 500;
  
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = model_input->data.int8;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;
}

// Interrupt Service Routine for handling voice commands
void voiceISR() {
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }
  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);
  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  RespondToCommand(error_reporter, current_time, found_command, score,
                   is_new_command);
}

// Pong functions
void drawField(int score1, int score2) {
    display.fillRect(0, round(paddlePositionPlayer1), 2, 18, 1);
    display.fillRect(126, round(paddlePositionPlayer2), 2, 18, 1);
  
    display.setTextSize(1);
    display.setTextColor(WHITE);
    display.setCursor(55, 0);
    display.print(score1);
    display.print(":");
    display.print(score2);

    display.fillRect(63, 12, 1, 5, 1);
    display.fillRect(63, 22, 1, 5, 1);
    display.fillRect(63, 32, 1, 5, 1);
    display.fillRect(63, 42, 1, 5, 1);
    display.fillRect(63, 52, 1, 5, 1);
    display.fillRect(63, 62, 1, 5, 1);
}

void collisionControl() {
  //bounce from top and bottom
  if (ballY >= 64 - 2 || ballY <= 0) {
    ballSpeedY *= -1;
  } 

  //score points if ball hits wall behind player
  if (ballX >= 128 - 2 || ballX <= 0) {
    if (ballSpeedX > 0) {
      scorePlayer1++;
      ballX = 128 / 4;
    }
    if (ballSpeedX < 0) {
      scorePlayer2++;
      ballX = 128 / 4 * 3;
    }  
    if (scorePlayer1 == 2 || scorePlayer2 == 2) {
      gameState = 2;
    }
  }

  //bounce from player1
  if (ballX >= 0 && ballX <= 2 && ballSpeedX < 0) {
    if (ballY > round(paddlePositionPlayer1) - 2 && ballY < round(paddlePositionPlayer1) + 18) {
      ballSpeedX *= -1;
    }
  }
  //bounce from player2
  if (ballX >= 128-2-2 && ballX <= 128-2 && ballSpeedX > 0) {
    if (ballY > round(paddlePositionPlayer2) - 2 && ballY < round(paddlePositionPlayer2) + 18) {
      ballSpeedX *= -1;
    }

  }
}

void drawBall() {
  display.fillRect(ballX, ballY, 2, 2, 1);
  
  ballX += ballSpeedX;
  ballY += ballSpeedY;
}

void loop() {
  // Get current input command
  voiceISR();
  
  // if voiceISR() == 'up', 'down', 'start', or other, set display as appropropiate
  if (pong_command == 1 && gameState == 0) {
    gameState = 1;
    pong_command = 0;
  }
  else if (pong_command == 1) {
    controllerValue1 += 200;
    controllerValue2 -= 200;
    pong_command = 0;
  }
  else if (pong_command == 2) {
    controllerValue1 -= 200;
    controllerValue2 += 200;
    pong_command = 0;
  }

  paddlePositionPlayer1 = controllerValue1 * (46.0 / 1023.0);
  paddlePositionPlayer2 = controllerValue2 * (46.0 / 1023.0);

  if (buttonState == HIGH && gameState == 0) {
      gameState = 1;
      delay(100);
  } else if (buttonState == HIGH && (gameState == 1 || gameState == 2)) {
      gameState = 0;
      scorePlayer1 = 0;
      scorePlayer2 = 0;
      ballX = 128/2;
      ballY = 64/2;
      delay(100);
  }
  
  if (gameState == 0) {
    display.setTextSize(2);
    display.setTextColor(WHITE);
    display.setCursor(40, 18);
    display.println("PONG");
    display.setTextSize(1);
    display.setTextColor(WHITE);
    display.setCursor(32, 38);
    display.println("say start");
    display.display();
    display.clearDisplay();      
  }

  if (gameState == 1) {
    drawField(scorePlayer1, scorePlayer2);
    
    collisionControl();
    drawBall();
    
    display.display();
    display.clearDisplay();
  }

  if (gameState == 2) {
    drawField(scorePlayer1, scorePlayer2);

    display.setTextSize(1);
    display.setTextColor(WHITE);
    
    if (scorePlayer1 == 2) {
      display.setCursor(15, 30);
    } else if (scorePlayer2 == 2) {
      display.setCursor(77, 30);
    }
    display.println("winner!");
    display.display();
    display.clearDisplay();   
  }
}
