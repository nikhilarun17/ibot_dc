/*
Developer   - Nikhil Arun
Date        - 19-2-2026
Board       - ESP32
*/

// blinking the built-in LED 
#define LED_BUILTIN 2   

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);  // LED ON
  delay(1000);                  
  digitalWrite(LED_BUILTIN, LOW);   // LED OFF
  delay(1000);                      
}
