/*
Developer   - Nikhil Arun
Date        - 3-2-2026
Board       - Arduino Uno R3
*/

// When detecting white object displays a value of arond 34-35 and otherwise displays 1021

int irpin = A0;
void setup() {
  Serial.begin(9600);
}

void loop() {
  float irvalue = analogRead(irpin);
  Serial.println(irvalue);
}
