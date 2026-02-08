/*
Developer   - Nikhil Arun
Date        - 7-2-2026
Board       - Arduino Uno R3
*/

// Moves a servo motor to and fro

#include <Servo.h>

int potpin = 11;
Servo My_Servo;

void setup() {
  My_Servo.attach(potpin);
}

void loop() {
  for (int i=0;i<255;i++){
      My_Servo.write(i);
        delay(10);

  }
  for (int i=255;i>0;i--){
      My_Servo.write(i);
        delay(10);

  }
  delay(20);
  }
