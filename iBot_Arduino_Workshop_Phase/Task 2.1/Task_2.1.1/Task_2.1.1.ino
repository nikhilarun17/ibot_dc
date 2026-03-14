/*
Developer   - Nikhil Arun
Date        - 19-2-2026
Board       - ESP32
*/

// connecting to WiFi network using ESP32 and printing local IP address

#include <WiFi.h>

char* ssid = "vivo X200 FE";
char* password = "Example@009";

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Hi,this is nikhil");

  WiFi.begin(ssid,password);

  while (WiFi.status() != WL_CONNECTED ){
    Serial.print(".");
  }
  Serial.println("Wifi connected");
  Serial.println(WiFi.localIP());
}


void loop() {
}
