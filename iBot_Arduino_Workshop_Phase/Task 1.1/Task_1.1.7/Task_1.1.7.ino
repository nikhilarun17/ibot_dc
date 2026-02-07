/*
Developer   - Nikhil Arun
Date        - 3-2-2026
Board       - Arduino Uno R3
*/

// Uses the DHT11 sensor to detect the humidity and temperature of the surroundings. Blowing air on top of the sensor increases the humidity of the space and the detection values too.
// Incorporates the TinyDHT arduino library to do all.

#include <TinyDHT.h>

int DHTPin = 2;

DHT My_DHT(DHTPin, DHT11);

void setup() {
  Serial.begin(9600);
  My_DHT.begin();
}

void loop() {
  float humidity = My_DHT.readHumidity();
  float temperature = My_DHT.readTemperature(); 

  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.print(" %  |  Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");

  delay(2000);  
}
