/*
Developer   - Nikhil Arun
Date        - 3-2-2026
Board       - Arduino Uno R3
*/

// Used to detect motion (has a cooldown of about 8sec).. Displays motion detected if a slight hint of motion is detected.

int PIRpin = 6;
int PIRvalue =0;
void setup() {
  pinMode(PIRpin,INPUT);
  Serial.begin(9600);
}

void loop() {
  PIRvalue = digitalRead(PIRpin);
  if (PIRvalue == HIGH) {
  Serial.println("Motion detected");
  } else {
  Serial.println("No motion");
  }
  delay(100);
}
