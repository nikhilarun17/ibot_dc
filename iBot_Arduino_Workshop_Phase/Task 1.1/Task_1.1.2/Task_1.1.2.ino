/*
Developer   - Nikhil Arun
Date        - 3-2-2026
Board       - Arduino Uno R3
*/

// Displays 1023 when detecting darkness or else displays 34


int LDRPin = A0;
int LDRValue = 0;
void setup() {
  Serial.begin(9600);
}


void loop() {
  LDRValue = analogRead(LDRPin);
  Serial.println(LDRValue);
  delay(100);
}
