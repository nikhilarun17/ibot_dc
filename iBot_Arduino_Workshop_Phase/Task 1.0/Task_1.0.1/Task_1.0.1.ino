/*
Developer   - Nikhil Arun
Date        - 2-2-2026
Board       - Arduino Uno R3
*/

// Makes the builtin led blink 5 Hz

int LED_PIN=LED_BUILTIN;

void setup()
{
  pinMode(LED_PIN, OUTPUT);
}

void loop()
{
  digitalWrite(LED_PIN, HIGH);
  delay(200); 
  digitalWrite(LED_PIN, LOW);
  delay(200); 
}
