/*
Developer   - Nikhil Arun
Date        - 6-2-2026
Board       - Arduino Uno R3
*/

// Sets up a LCD display to display a ripple effect and hello world.

#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 2, 3, 4, 5);
void setup() {
  lcd.begin(16, 2);
  lcd.clear();
  lcd.print("Hello World");
  lcd.setCursor(0,1);
  lcd.print("Nikhil Arun");
}

void loop() {}
