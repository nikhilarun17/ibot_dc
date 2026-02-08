/*
Developer   - Nikhil Arun
Date        - 7-2-2026
Board       - Arduino Uno R3
*/

// And OLED Snake-type game with a single pixel.

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

Adafruit_SSD1306 display(128,64,&Wire,-1);
int x_init = 0;
int y_init = 0;
int size = 8;
int x = x_init;
int y = y_init;

int left = 3;
int up = 4;
int down = 5;
int right = 6;

int up_value;
int down_value;
int left_value;
int right_value;

void setup() {
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    for(;;);
  }
  display.clearDisplay();
  display.setTextSize(1);

  pinMode(up,INPUT_PULLUP);
  pinMode(down,INPUT_PULLUP);
  pinMode(left,INPUT_PULLUP);
  pinMode(right,INPUT_PULLUP);
  delay(1000);
}

void loop() {
    display.clearDisplay();
    display.fillRect(x,y,size,size,SSD1306_WHITE);
    up_value = digitalRead(up);
    down_value = digitalRead(down);
    left_value = digitalRead(left);
    right_value = digitalRead(right);

    if (up_value){
      y-=size;
      if (y<=y_init-1){
        y=y_init+64-size;
      }
    }
    if (down_value){
      y+=size;
      if (y>=y_init+65-size){
        y=y_init;
      }
    }
    if (left_value){
      x-=size;
      if (x<=x_init-1) {
        x=x_init+128-size;
      }
    }
    if (right_value){
      x+=size;
      if (x>=x_init+128-size){
        x=x_init;
      }
    }
    delay(50);
    display.display();
   
}
