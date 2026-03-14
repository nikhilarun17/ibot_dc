/*
Developer   - Nikhil Arun
Date        - 10-3-2026
Board       - ESP32
*/

// ESP-NOW communication: Sending button states from master ESP32 to slave ESP32

#include <esp_now.h>
#include <WiFi.h>

uint8_t receiverMac[] = {0xC0, 0xCD, 0xD6, 0xCE, 0x91, 0x7C};

typedef struct buttoncollage {
  int button1;
  int button2;
  int button3;
  int button4;
} buttoncollage;

buttoncollage buttons;

void OnDataSent(const wifi_tx_info_t *info, esp_now_send_status_t status)
{
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Sent" : "Failed");
}
int b1 = 21;
int b2 = 19;
int b3 = 18;
int b4 = 4;

void setup() {

  Serial.begin(9600);

  pinMode(b1, INPUT_PULLUP);
  pinMode(b2, INPUT_PULLUP);
  pinMode(b3, INPUT_PULLUP);
  pinMode(b4, INPUT_PULLUP);

  WiFi.mode(WIFI_STA);
  
  esp_now_init();
  esp_now_register_send_cb(OnDataSent);
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  esp_now_add_peer(&peerInfo);
}

void loop() {

  buttons.button1 = !digitalRead(b1);
  buttons.button2 = !digitalRead(b2);
  buttons.button3 = !digitalRead(b3);
  buttons.button4 = !digitalRead(b4);

  Serial.printf("B1:%d B2:%d B3:%d B4:%d\n",
              buttons.button1,
              buttons.button2,
              buttons.button3,
              buttons.button4);

  esp_now_send(receiverMac, (uint8_t *) &buttons, sizeof(buttons));

  delay(50);
}
