/*
Developer   - Nikhil Arun
Date        - 3-2-2026
Board       - Arduino Uno R3
*/

// Uses the sound sensor and compares with a given threshold to detect whether a sound is a loud clap or not and displays the result.

int sound = A0;
int threshold = 200;
int LEDpin = LED_BUILTIN;
void setup() {
  Serial.begin(9600);
  pinMode(LEDpin, INPUT);
}

void loop() {
  float soundvalue = analogRead(sound);
  Serial.println(soundvalue);
  if (soundvalue > threshold) {
    digitalWrite(LEDpin,HIGH);
    Serial.println("LOUDSOUND");
    delay(2000);
  }
  delay(100);
}
