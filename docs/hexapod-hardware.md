# Hexapod Hardware Design

Four hexapod robots driven by simulated fruit fly brains (BANC connectome).
Brain runs on BC-250 GPUs, VNC (ventral nerve cord) runs on ESP32.

## Platform

**Base:** [SIXpack Robot](https://makerworld.com/en/models/1822096-sixpack-robot-sixlegged-fpv-walking-marvel) by delta3Robotics
- 12 servos (2 per leg: coxa yaw + femur lift)
- ESP32-CAM controller (we use ESP32-WROVER-DEV v1.6 for VNC sim)
- PCA9685 16-channel PWM driver over I2C
- 3D printed (PETG), ~€50 per bot
- Open-source firmware

## Servos

**SG90** (prototype) → **MG90S** (durable revision)

| | SG90 | MG90S |
|---|---|---|
| Gears | Plastic | Metal |
| Torque | 1.8 kg·cm | 2.2 kg·cm |
| Speed | 0.10s/60° | 0.10s/60° |
| Price | ~$2.50 | ~$4.50 |
| 12/bot | ~$30 | ~$55 |

Same speed — MG90S buys durability, not speed. Start with SG90.

**Estimated walking speed:** ~100 mm/s (0.1 m/s) with 2.5Hz tripod gait, 40mm stride.

## IR Beacon System (Charger Homing + Social ID)

### Concept
- **Charger:** IR LED blinking at 38kHz (TV-remote style), ID 0x00 = "food"
- **Each bot:** IR LED transmitting unique 8-bit ID (0x01–0x04)
- **Receivers:** 2× TSOP38238 in funnel shrouds, directional left/right

### Funnel Design
- 30° half-angle cone, 8mm deep
- 2mm bore at base → 10mm mouth
- ~90° FOV per sensor, ~60° overlap zone
- Both see beacon = go straight; left stronger = turn left
- Print mouth-up, no supports. Receiver press-fits at base.

See `ir-funnel-accurate.svg` for exact geometry.

### Why funnels, not tubes
Funnels reflect off-axis IR inward → more signal, wider FOV, longer range.
Tradeoff: less angular precision, but homing only needs left/right.

### Why not a camera for IR
IR beacon wants IR-pass filter (blocks visible). Visual neurons need visible light.
Different filters, different exposures — can't multiplex. Use both:
- **Camera** (visible): visual neurons, sees other bots, obstacles
- **IR receivers** (analog): fast beacon homing, no image processing

### Parts per bot
| Part | Qty | Cost |
|---|---|---|
| TSOP38238 receiver | 2 | ~$1 |
| IR LED 940nm | 1 | ~$0.20 |
| 2N2222 transistor | 1 | ~$0.10 |
| 100Ω resistor | 1 | ~$0.05 |

## Current Sensing (Obstacle Detection)

**INA219** I2C current sensor per leg (6 total, ~$6/bot).

- Normal walking: ~300-500mA per leg (2× SG90)
- Obstacle contact: spikes to 800mA+
- Threshold: >2× running baseline for >100ms = contact
- Maps to campaniform sensilla (fly load sensors) → VNC reflex

**I2C addressing:** INA219 supports 4 addresses. Use 2 I2C buses (3 sensors each).
Alternative: 2× INA3221 (triple-channel) = 2 chips for 6 legs.

## ESP32-WROVER-DEV v1.6 Pin Map

### I2C Bus 0 (GPIO 21=SDA, 22=SCL)
| Device | Address | Purpose |
|---|---|---|
| PCA9685 | 0x40 | 12 servo PWM |
| INA219 | 0x41, 0x44, 0x45 | Legs 0-2 current |

### I2C Bus 1 (GPIO 32=SDA, 33=SCL)
| Device | Address | Purpose |
|---|---|---|
| INA219 | 0x41, 0x44, 0x45 | Legs 3-5 current |

### GPIO
| Pin | Dir | Purpose | Notes |
|---|---|---|---|
| 2 | Out | Heartbeat LED | Onboard, strapping-safe |
| 13 | In | IR RX left | Free |
| 14 | In | IR RX right | Free |
| 18 | Out | IR TX (via transistor) | Moved from 15 (strapping) |
| 34 | In (ADC) | Battery voltage | Input-only, ideal for ADC |
| 1/3 | — | USB serial | Reserved |

**Avoid:** GPIO 6-11 (flash/PSRAM), GPIO 15 (strapping — do not use for IR TX).

### Core Allocation
- **Core 0:** VNC neural sim (5K quantized neurons, 1ms timestep, INT8 weights in PSRAM)
- **Core 1:** I/O loop — sensors → ESP-NOW → brain, DN rates → servos (50Hz)

## Brain ↔ VNC Protocol (ESP-NOW)

### Brain (BC-250) → VNC (ESP32), 50Hz
| Signal | Type | Meaning |
|---|---|---|
| DNp09 rate | float | Walk forward intent |
| DNa02 L/R | 2× float | Turn intent |
| MDN rate | float | Walk backward intent |

~16 bytes = 0.8 KB/s. Descending neuron rates are the anatomical interface.

### VNC (ESP32) → Brain (BC-250), 50Hz
| Signal | Type | Meaning |
|---|---|---|
| IR L/R | 2× uint8 | Beacon tracking |
| Bot IDs + bearings | ~8 bytes | Social |
| Camera (30×30×2) | 1800 bytes @ 20Hz | Vision (photoreceptor inputs) |
| Servo positions | 12× uint8 | Proprioception |
| Battery | uint16 | Hunger |
| CS load | 6× uint8 | Obstacle contact per leg |

~37 KB/s per bot. 4 bots = 150 KB/s = 1.2 Mbps (WiFi handles easily).

### Failsafe
Radio drop → VNC holds last DN rates 200ms, then decays to zero (stop, don't fall).
Brain continues; resumes on reconnect.

## Architecture Summary

```
┌─────────────┐     ESP-NOW      ┌──────────────┐
│  BC-250     │ ◄──────────────► │  ESP32       │
│  (Brain)    │   DN rates /     │  (VNC)       │
│  156K       │   sensors        │  5K neurons  │
│  neurons    │                  │  (quantized) │
│  FP32       │                  │  INT8        │
└─────────────┘                  └──────┬───────┘
                                       │ I2C/GPIO
                              ┌────────┴────────┐
                              │  PCA9685,       │
                              │  INA219×6,      │
                              │  IR, servos     │
                              └─────────────────┘
```

Brain decides *where* (navigation). VNC decides *how* (walking).
Same as fly anatomy.
