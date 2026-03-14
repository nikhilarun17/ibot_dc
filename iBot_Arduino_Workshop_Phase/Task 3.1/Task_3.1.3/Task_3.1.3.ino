/*
Developer   - Nikhil Arun
Date        - 10-3-2026
Board       - ESP32
*/

// ESP-NOW communication: Receiving button states on slave ESP32 and controlling motors (LED) accordingly

#include <esp_now.h>
#include <WiFi.h>

typedef struct buttoncollage {
  int button1;
  int button2;
  int button3;
  int button4;
} buttoncollage;

buttoncollage myData;

void control()
{
    // STOP (no buttons)
    if(myData.button1==0 && myData.button2==0 && myData.button3==0 && myData.button4==0)
    {
        digitalWrite(33,LOW);
        digitalWrite(25,LOW);
        digitalWrite(26,LOW);
        digitalWrite(27,LOW);
    }

    // FORWARD
    else if(myData.button1==1)
    {
        digitalWrite(33,HIGH); // LMF
        digitalWrite(25,LOW);  // LMB
        digitalWrite(26,HIGH); // RMF
        digitalWrite(27,LOW);  // RMB
    }

    // BACKWARD
    else if(myData.button2==1)
    {
        digitalWrite(33,LOW);
        digitalWrite(25,HIGH);
        digitalWrite(26,LOW);
        digitalWrite(27,HIGH);
    }

    // LEFT (Anti-clockwise spin)
    else if(myData.button3==1)
    {
        digitalWrite(33,LOW);
        digitalWrite(25,HIGH);
        digitalWrite(26,HIGH);
        digitalWrite(27,LOW);
    }

    // RIGHT (Clockwise spin)
    else if(myData.button4==1)
    {
        digitalWrite(33,HIGH);
        digitalWrite(25,LOW);
        digitalWrite(26,LOW);
        digitalWrite(27,HIGH);
    }
}
void OnDataRecv(const esp_now_recv_info *info, const uint8_t *incomingData, int len) {
    memcpy(&myData, incomingData, sizeof(myData));
    control();
    Serial.printf("B1:%d B2:%d B3:%d B4:%d\n",
              myData.button1,
              myData.button2,
              myData.button3,
              myData.button4);
    Serial.println("Packet received");

}

void setup()

{
Serial.begin(9600);
    WiFi.mode(WIFI_STA);
    pinMode(33,OUTPUT);
    pinMode(27,OUTPUT);
    pinMode(25,OUTPUT);
    pinMode(26,OUTPUT);

    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW");
        return;
    }

    esp_now_register_recv_cb(OnDataRecv);
}

void loop()
{
}

\
