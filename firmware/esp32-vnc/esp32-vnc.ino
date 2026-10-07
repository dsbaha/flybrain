/*
 * FlyBrain VNC Firmware — ESP32-WROVER-DEV v1.6
 *
 * Phase 1 (this file): Board validation skeleton.
 *  - Serial protocol (115200 baud)
 *  - PWM test outputs (servo signal simulation)
 *  - Camera stub (for ESP32-CAM variant / external OV2640)
 *  - Heartbeat LED
 *
 * Phase 2 (next): Quantized VNC leg-control subcircuit on Core 0,
 *  I/O + ESP-NOW on Core 1.
 *
 * Phase 3: Full DN protocol with BC-250 brain server.
 */

#include <Arduino.h>

// ── Pin assignments (ESP32-WROVER-DEV v1.6) ──
#define LED_PIN        2    // Built-in LED (heartbeat)
#define SERVO_TEST_PIN 13   // PWM test output (scope/logic analyzer here)
#define IR_TX_PIN      14   // IR LED transmit (bot ID beacon) — future
#define IR_RX_LEFT     32   // IR receiver left — future
#define IR_RX_RIGHT    33   // IR receiver right — future

// ── Servo PWM config ──
// SG90: 50Hz, 1ms = 0°, 1.5ms = 90°, 2ms = 180°
#define SERVO_FREQ     50
#define SERVO_RES      16   // 16-bit resolution (0-65535)
#define SERVO_CH       0

// ── Protocol (serial, 115200) ──
// Commands (host → ESP32):
//   PING          → PONG
//   SERVO <ch> <us>  → set servo pulse width (500-2500us)
//   SWEEP         → sweep test servo 0°→180°→0°
//   STATUS        → report firmware state
// Telemetry (ESP32 → host, 50Hz):
//   T,<millis>,<servo_us>,<ir_l>,<ir_r>,<vbat_mv>

static uint32_t lastTelemetry = 0;
static uint32_t lastHeartbeat = 0;
static bool ledState = false;
static int servoPulseUs = 1500;  // center
static bool sweeping = false;
static int sweepDir = 1;

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("# FlyBrain VNC firmware v0.1.0");
  Serial.println("# Board: ESP32-WROVER-DEV v1.6");
  Serial.println("# Cores: 2 @ 240MHz");

  // Report memory
  Serial.printf("# SRAM free: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("# PSRAM free: %d bytes\n", ESP.getFreePsram());
  Serial.printf("# Flash: %d bytes\n", ESP.getFlashChipSize());

  // LED
  pinMode(LED_PIN, OUTPUT);

  // Servo PWM (LEDC)
  ledcSetup(SERVO_CH, SERVO_FREQ, SERVO_RES);
  ledcAttachPin(SERVO_TEST_PIN, SERVO_CH);
  setServoUs(servoPulseUs);

  // IR pins (future — just configure for now)
  pinMode(IR_TX_PIN, OUTPUT);
  digitalWrite(IR_TX_PIN, LOW);
  pinMode(IR_RX_LEFT, INPUT);
  pinMode(IR_RX_RIGHT, INPUT);

  Serial.println("# READY");
  Serial.println("# Commands: PING, SERVO <us>, SWEEP, STATUS");
}

void setServoUs(int us) {
  // Convert microseconds to 16-bit duty (50Hz = 20000us period)
  us = constrain(us, 500, 2500);
  uint32_t duty = (uint32_t)us * 65535 / 20000;
  ledcWrite(SERVO_CH, duty);
  servoPulseUs = us;
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd == "PING") {
    Serial.println("PONG");
  } else if (cmd.startsWith("SERVO ")) {
    int us = cmd.substring(6).toInt();
    setServoUs(us);
    sweeping = false;
    Serial.printf("OK servo=%dus\n", servoPulseUs);
  } else if (cmd == "SWEEP") {
    sweeping = true;
    Serial.println("OK sweeping");
  } else if (cmd == "STATUS") {
    Serial.printf("STATUS fw=0.1.0 heap=%d psram=%d servo=%d sweep=%d\n",
      ESP.getFreeHeap(), ESP.getFreePsram(), servoPulseUs, sweeping);
  } else if (cmd.length() > 0) {
    Serial.printf("ERR unknown: %s\n", cmd.c_str());
  }
}

void loop() {
  uint32_t now = millis();

  // ── Heartbeat LED (1Hz) ──
  if (now - lastHeartbeat > 500) {
    lastHeartbeat = now;
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
  }

  // ── Serial commands ──
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }

  // ── Servo sweep test ──
  if (sweeping) {
    servoPulseUs += sweepDir * 10;
    if (servoPulseUs >= 2500) { servoPulseUs = 2500; sweepDir = -1; }
    if (servoPulseUs <= 500)  { servoPulseUs = 500;  sweepDir = 1; }
    setServoUs(servoPulseUs);
    delay(20);
  }

  // ── Telemetry (50Hz) ──
  if (now - lastTelemetry > 20) {
    lastTelemetry = now;
    int irL = digitalRead(IR_RX_LEFT);
    int irR = digitalRead(IR_RX_RIGHT);
    // VBAT: stub — wire ADC later
    int vbatMv = 0;
    Serial.printf("T,%lu,%d,%d,%d,%d\n", now, servoPulseUs, irL, irR, vbatMv);
  }
}

// ── Phase 2 notes ──
// Core 0: quantized VNC LIF sim (5K neurons, 1ms timestep, INT8 weights in PSRAM)
// Core 1: this I/O loop (sensors → ESP-NOW → brain, DN rates → VNC → servos)
// Interface: DNp09/DNa02/MDN firing rates via shared memory (no locks, single writer)
