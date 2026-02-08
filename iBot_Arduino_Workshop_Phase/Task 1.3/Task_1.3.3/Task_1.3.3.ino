#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#define N 16

Adafruit_SSD1306 display(128,64,&Wire,-1);
int sound_value = 0;
int arr[N] = {0};

#include <stdio.h>


void shift_and_insert(int arr[], int size, int new_val) {
    for (int i = size - 1; i > 0; i--) {
        arr[i] = arr[i - 1];
    }
    arr[0] = new_val;
}


void setup() {
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    for(;;);
  }
  display.clearDisplay();
  display.setTextSize(1);
  pinMode(A0,INPUT);
  Serial.begin(9600);
}

void loop() {
  sound_value = analogRead(A0);
  sound_value = (sound_value*252)/1023;
  shift_and_insert(arr,N,sound_value);
  display.clearDisplay();
  for (int i = 0; i<N;i++){
    display.fillRect(8*i,48-arr[i],8,arr[i],SSD1306_WHITE);
  }
  display.display();
  delay(200);
  Serial.println(sound_value);
  
}
