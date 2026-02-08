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
