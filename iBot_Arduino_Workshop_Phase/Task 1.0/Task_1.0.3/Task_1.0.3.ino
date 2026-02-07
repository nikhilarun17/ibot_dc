/*
Developer   - Nikhil Arun
Date        - 2-2-2026
Board       - Arduino Uno R3
*/

// Uses a pushbutton as a switch to turn the LED on and off. Holding the Switch doesnt result in it blinking.

int PUSHPIN =2;
int LEDPIN = 3;
int state =1;
int previoustate=0;


void setup() {
  pinMode(PUSHPIN, INPUT_PULLUP);
  pinMode(LEDPIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  int buttonstate = digitalRead(PUSHPIN);
  if((not buttonstate) and (previoustate==HIGH)){
  state=1-state;
  }
  Serial.println(state);
  if (state){
    digitalWrite(LEDPIN,HIGH);
  }
  else {
    digitalWrite(LEDPIN,LOW);
  }
  
  previoustate=buttonstate;
  delay(200);
}
