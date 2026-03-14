/*
Developer   - Nikhil Arun
Date        - 19-2-2026
Board       - ESP32
*/

// Creating a simple web server on ESP32 to control the built-in LED

#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "vivo X200 FE";
const char* password = "Example@009";

WebServer server(80);

#define LED_PIN 2 

void handleRoot() {
  String html = "<!DOCTYPE html><html>";
  html += "<body>";
  html += "<h1>ESP32 Control</h1>";
  html += "<a href=\"/on\"><button>LED ON</button></a>";
  html += "<a href=\"/off\"><button>LED OFF</button></a>";
  html += "</body></html>";

  server.send(200, "text/html", html);
}

void handleOn() {
  digitalWrite(LED_PIN, HIGH);
  server.send(200, "text/plain", "LED is ON");
}

void handleOff() {
  digitalWrite(LED_PIN, LOW);
  server.send(200, "text/plain", "LED is OFF");
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);

  WiFi.begin(ssid, password);
  Serial.print("Connecting");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/on", handleOn);
  server.on("/off", handleOff);

  server.begin();
  Serial.println("Server started");
}

void loop() {
  server.handleClient();
}
