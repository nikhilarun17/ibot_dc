/*
Developer   - Nikhil Arun
Date        - 19-2-2026
Board       - ESP32
*/

// breathing LED effect using PWM on ESP32
int pwm_frequency = 5000;
int pwm_resolution = 10;
int pwm_pin =15;

unsigned long prevmill = 0;
const int interval = 2;  

int brightness = 0;
int fade = 1;

void setup() {
  ledcAttach(pwm_pin,pwm_frequency, pwm_resolution);
}

void loop() {
  unsigned long mill = millis();
  if ((mill - prevmill)>=interval) {
    prevmill = mill;
    ledcWrite(pwm_pin, brightness);

    brightness += fade;

    if (brightness <= 0 || brightness >= 1023) {
      fade = -fade;  // fade direction
  }}
}
