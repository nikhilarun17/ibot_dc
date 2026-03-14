/*
Developer   - Nikhil Arun
Date        - 10-3-2026
Board       - ESP32
*/

// Bluetooth control of an LED on ESP32 using BluetoothSerial library

#include <BluetoothSerial.h>
int ledpin = 18;
BluetoothSerial SerialBT;

void setup() {
  Serial.begin(9600);
    SerialBT.begin("ESP32_Nikhil");
    pinMode(ledpin, OUTPUT);
}

void loop() {
  if (SerialBT.available()) {
    Serial.println("Recieving");
    }
    char val = SerialBT.read();
    if (val=='1'){
      digitalWrite(ledpin, HIGH);
    }
    if (val=='0'){
      digitalWrite(ledpin, LOW);
    }
    delay(1000);
}
