#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

Adafruit_SSD1306 display(128,64,&Wire,-1);


void setup() {
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    for(;;);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(10, 10);
  display.println("Hello, World!");
  display.display();
  delay(1000);
}


void loop() {
  display.clearDisplay();

for (float r = 2; r <= 15; r++) {
  display.clearDisplay();
  display.display();
  display.drawCircle(80, 32, r, WHITE);
  display.display();
  delay(1);
  display.drawCircle(20, 32, r-5, WHITE);
  display.display();
  display.drawCircle(100, 40, r-5, WHITE);
  display.display();
  delay(25);
  display.drawCircle(55, 20, r-5, WHITE);
  display.display();
  delay(25);
}
}
