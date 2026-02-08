/*
Developer   - Nikhil Arun
Date        - 7-2-2026
Board       - Arduino Uno R3
*/

// Uses a lazer and when interrupted starts a buzzer and displays it on the lcd screen

#include <LiquidCrystal.h>
int ldr = 0;
LiquidCrystal lcd(12, 11, 2, 3, 4, 5);
void setup() {
 pinMode(6,OUTPUT);
 pinMode(9,INPUT);
 lcd.begin(16, 2);

}

void loop() {
  ldr = digitalRead(9);
  if (ldr==0){
    noTone(6);
    lcd.print("No Interference");
  }
  else{
    tone(6,200);
    lcd.print("Interference");
    lcd.setCursor(5,1);
    lcd.print("Detected");
  }
  delay(400);
    lcd.clear();

}
