/*
Developer   - Nikhil Arun
Date        - 10-3-2026
Board       - ESP32
*/

// Connecting to WiFi network using ESP32 and printing MAC address

#include <WiFi.h>

const char* ssid = "Mi 11X";
const char* password = "Nharen07";

void setup() {
  Serial.begin(9600);
  delay(1000);
  Serial.println("Hi,this is nikhil");

  WiFi.begin(ssid,password);

  while (WiFi.status() != WL_CONNECTED ){
    Serial.print(".");
  }
  Serial.println("Wifi connected");
  Serial.println(WiFi.macAddress());
}


void loop() {
  // put your main code here, to run repeatedly:

}
