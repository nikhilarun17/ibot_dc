/*
Developer   - Nikhil Arun
Date        - 7-3-2026
Board       - Arduino Uno R3
*/

#include <Wire.h>
#include <Servo.h>

Servo myservo;
Servo myservo2;
Servo myservo3;

int buzzer = 2;

int led1 = 3;
int led2 = 4;
int led3 = 5;
int led4 = 6;

unsigned long lastServoTime = 0;
unsigned long lastBuzzerTime = 0;
unsigned long lastLEDTime = 0;

unsigned long servoInterval = 150;
unsigned long buzzerInterval;
unsigned long ledInterval;

bool servoState = true;
bool systemOn = true;

void receiveEvent(int bytes) {
  if (Wire.available()) {
    systemOn = Wire.read();
  }
}

void setup() {

  Wire.begin(8);
  Wire.onReceive(receiveEvent);

  myservo.attach(9);
  myservo2.attach(10);
  myservo3.attach(11);
  myservo.write(0);
  myservo2.write(0);
  myservo3.write(0);

  pinMode(buzzer, OUTPUT);

  pinMode(led1, OUTPUT);
  pinMode(led2, OUTPUT);
  pinMode(led3, OUTPUT);
  pinMode(led4, OUTPUT);

  randomSeed(analogRead(A0));

  buzzerInterval = random(50, 200);
  ledInterval = random(30, 120);
}

void loop() {

  if (systemOn) {

    unsigned long currentTime = millis();

    // SERVO CONTROL
    if (currentTime - lastServoTime >= servoInterval) {
      lastServoTime = currentTime;

      if (!servoState) {
        myservo.write(180);
        myservo2.write(0);
        myservo3.write(0);
        servoState = true;
      } else {
        myservo.write(0);
        myservo2.write(180);
        myservo3.write(180);
        servoState = false;
      }
    }

    // BUZZER CONTROL
    if (currentTime - lastBuzzerTime >= buzzerInterval) {
      lastBuzzerTime = currentTime;
      buzzerInterval = random(50, 200);

      int freq = random(500, 4000);
      tone(buzzer, freq, random(30, 100));
    }

    // RANDOM LED FLASHING
    if (currentTime - lastLEDTime >= ledInterval) {
      lastLEDTime = currentTime;
      ledInterval = random(30, 120);

      digitalWrite(led1, random(0,2));
      digitalWrite(led2, random(0,2));
      digitalWrite(led3, random(0,2));
      digitalWrite(led4, random(0,2));
    }
  }
  if (not systemOn){
    digitalWrite(led1, 0);
      digitalWrite(led2, 0);
      digitalWrite(led3, 0);
      digitalWrite(led4,0);
      myservo.write(0);
      myservo2.write(0);
      myservo3.write(0);
      
  }
  
  
  
  }
