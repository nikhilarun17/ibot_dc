/*
Developer   - Nikhil Arun
Date        - 2-2-2026
Board       - Arduino Uno R3
*/

//LED brightness increases and diminishes in 1s

int LED_PIN=6;

void setup()
{
  pinMode(LED_PIN, OUTPUT);
}

void loop()
{
  for (int i=0;i<256;i++)
  {
    analogWrite(LED_PIN, i);
    delay(500/256);
  }
  for (int j=0;j<256;j++)
  {
    analogWrite(LED_PIN, 255-j);
    delay(500/256);
  }
}
