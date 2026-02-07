/*
Developer   - Nikhil Arun
Date        - 3-2-2026
Board       - Arduino Uno R3
*/

// Used to send ultrasonic waves and uses the echo duration to detect the distance of the object. (Uses PulseIn to measure duration)

int trigpin = 9;
int echopin = 8;

float Distance, Duration;

void setup()
{
  pinMode(trigpin, OUTPUT);
  pinMode(echopin, INPUT);
  Serial.begin(9600);

}

void loop()
{
  digitalWrite(trigpin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigpin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigpin, LOW);

  Duration = pulseIn(echopin, HIGH, 30000);
  Distance = (Duration*0.0343)/2;

  if(Duration == 0)
  {
    Serial.println("No Object Detected");
    
  }
  else
  {
    Serial.print("Distance : ");
    Serial.print(Distance);
    Serial.print(" cm");
    Serial.print(" (Duration : ");
    Serial.print(Duration);
    Serial.println(" µs)");
    

  }
  delay(500);
  
}
