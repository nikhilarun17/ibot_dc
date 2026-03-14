/*
Developer   - Nikhil Arun
Date        - 7-3-2026
Board       - Arduino Uno R3
*/

#include <Wire.h>
#include <Servo.h>

Servo masterServo;

int buttonPin = 3;
int ldrPin = A0;
int ldrtanay = A1;

int threshold = 200;

bool slaveState = false;

bool servoState = false;

unsigned long lastServoTime = 0;
unsigned long servoInterval = 150;

unsigned long bufferStart = 0;
bool bufferActive = false;

int lastButtonState = HIGH;

void setup() {

  Serial.begin(9600);

  Wire.begin();

  pinMode(buttonPin, INPUT_PULLUP);

  masterServo.attach(9);
  masterServo.write(0);
}

void loop() {

  int buttonState = digitalRead(buttonPin);

  // BUTTON PRESSED → start 3s buffer
  if (buttonState == LOW && lastButtonState == HIGH) {

    bufferActive = true;
    bufferStart = millis();

    Serial.println("Button pressed → 3s buffer");

    delay(200);
  }

  lastButtonState = buttonState;

  // Buffer period
  if (bufferActive) {

    if (millis() - bufferStart >= 10000) {
      bufferActive = false;
      Serial.println("Laser detection active");
    } 
    else {
      return;
    }
  }

  int ldrValue = analogRead(ldrPin);

  Serial.print("LDR: ");
  Serial.println(ldrValue);

  unsigned long currentTime = millis();

  // LASER BROKEN
  if (ldrValue > threshold) {

    if (!slaveState) {

      slaveState = true;

      Serial.println("Laser broken → START SLAVE");

      Wire.beginTransmission(8);
      Wire.write(1);
      Wire.endTransmission();
    }

    // SERVO LOOP
    if (currentTime - lastServoTime >= servoInterval) {

      lastServoTime = currentTime;

      if (!servoState) {
        masterServo.write(180);
        servoState = true;
      } 
      else {
        masterServo.write(0);
        servoState = false;
      }
    }
  }

  // LASER RESTORED
  else {

    if (slaveState) {

      slaveState = false;

      Serial.println("Laser restored → STOP SLAVE");

      Wire.beginTransmission(8);
      Wire.write(0);
      Wire.endTransmission();

      masterServo.write(0);  // stop servo
    }
  }

  delay(50);
}
