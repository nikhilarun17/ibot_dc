/*
Developer   - Nikhil Arun
Date        - 6-2-2026
Board       - Arduino Uno R3
*/

//Makes the buzzer buzz with a certain frequency every 1 sec

int buzzerpin = 9;
void setup() {
  pinMode(buzzerpin, OUTPUT);
  noTone(buzzerpin);
}

void loop() {
  tone(buzzerpin,200);
  delay(1000);
    noTone(buzzerpin);
delay(1000);

}
