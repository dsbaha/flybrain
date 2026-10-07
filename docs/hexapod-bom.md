# Hexapod BOM — Fly-Brain Robot

Four autonomous brain-driven hexapod robots with social interaction.
3D printed SIXpack chassis, ESP32 VNC controller, BANC brain on BC-250.

**3D model:** [SIXpack Robot by delta3Robotics](https://makerworld.com/en/models/1822096-sixpack-robot-sixlegged-fpv-walking-marvel)
**Filament:** PETG, ~229g per bot (fits Bambu A1 mini bed)

---

## Design Tiers

| Tier | Per bot | 4 bots | What's included |
|---|---|---|---|
| **Classroom** | ~$44 | ~$176 | Brain→legs walking demo, USB powered |
| **Autonomous + Social** | ~$100 | ~$440 | Full build: battery, vision, IR social, touch sensing |

This BOM covers the **Autonomous + Social** tier (the complete design).
Classroom tier details at the bottom.

---

## Social Experiment Design

Each bot has a unique IR ID. They detect, identify, and interact.

### Individual Identity
| Bot | IR ID | Behavior bias |
|---|---|---|
| Bot 1 | 0x01 | Explorer (seeks novelty) |
| Bot 2 | 0x02 | Follower (attracted to others) |
| Bot 3 | 0x03 | Avoider (maintains distance) |
| Bot 4 | 0x04 | Forager (seeks charger) |
| Charger | 0x00 | "Food" — all bots home when hungry |

### Social Behaviors (brain-driven)
1. **Detection:** IR receivers pick up another bot's ID + bearing (L/R)
2. **Vision:** Camera sees other bots as moving objects → optic lobe
3. **Decision:** Brain weighs social signals vs. hunger vs. curiosity
4. **Action:** Approach, follow, avoid, or ignore — via DNp09/DNa02

### Experiments to Run
- **Aggregation:** Do bots cluster? (fly social behavior)
- **Charger competition:** 4 bots, 1 charger — who gets there first?
- **Leader/follower:** Does Bot 2 actually follow Bot 1?
- **Avoidance:** Does Bot 3 maintain personal space?

---

## Electronics — Per Bot (Autonomous + Social)

| # | Part | Qty | Unit $ | Source | Notes |
|---|---|---|---|---|---|
| 1 | ESP32-WROVER-DEV v1.6 | 1 | $8 | Amazon / AliExpress | Dev board w/ PSRAM. You may already have these. |
| 2 | PCA9685 16-ch PWM driver | 1 | $3 | Amazon / AliExpress | I2C servo driver, 0x40 |
| 3 | SG90 micro servo | 12 | $2.50 | Amazon (12-pack ~$25) | Start here. Upgrade to MG90S later. |
| 4 | INA219 current sensor | 6 | $1 | Amazon / AliExpress | Per-leg obstacle detection. Or 2× INA3221. |
| 5 | TSOP38238 IR receiver | 2 | $0.50 | Amazon / AliExpress | 38kHz, in funnel shrouds |
| 6 | IR LED 940nm 5mm | 2 | $0.10 | Amazon / AliExpress | 1 for TX, 1 spare |
| 7 | 2N2222 NPN transistor | 2 | $0.10 | Amazon / AliExpress | IR LED driver + spare |
| 8 | LM2596 buck converter | 1 | $1.50 | Amazon / AliExpress | 7.4V → 5V for servos (3A) |
| 9 | 2S 1000mAh LiPo (7.4V) | 1 | $8 | Amazon (Zeee, Ovonic) | ~20 min walk time |
| 10 | LiPo voltage alarm / low-voltage cutoff | 1 | $2 | Amazon | Protects battery |
| 11 | OV2640 camera module | 1 | $6 | Amazon / AliExpress | For ESP32 visual input |
| 12 | M3 brass standoffs kit | 1 | $5 | Amazon | Board stacking |
| 13 | M2 screws (assorted) | 1 pack | $5 | Amazon | Servo mounting (shared across bots) |
| 14 | Jumper wires (M-F, 20cm) | 1 pack | $4 | Amazon | (shared across bots) |
| 15 | 100Ω, 10kΩ resistors | 1 pack | $3 | Amazon | (shared across bots) |
| 16 | Micro USB cable | 1 | $3 | Amazon | Programming |

**Per-bot electronics subtotal: ~$75** (with SG90s)

### With MG90S upgrade (metal gear servos)

| # | Part | Qty | Unit $ | Notes |
|---|---|---|---|---|
| 3b | MG90S micro servo | 12 | $4.50 | Metal gears, same speed |

**Per-bot electronics subtotal: ~$100** (with MG90S)

---

## 3D Printed Parts — Per Bot

| Part | Source | Filament | Print time |
|---|---|---|---|
| SIXpack chassis + legs | MakerWorld link above | PETG ~229g | ~12h |
| IR funnel shrouds × 2 | `ir-funnel-accurate.svg` (convert to STL) | PETG ~5g | ~30min |
| Camera mount | SIXpack head (included) | — | — |

**Filament cost per bot:** ~$6 (PETG @ ~$25/kg)

---

## Tools Needed (one-time)

| Tool | Source | $ |
|---|---|---|
| Soldering iron + solder | Amazon | $15 |
| Wire cutters / flush cutters | Amazon | $8 |
| Small Phillips screwdriver | — | $5 |
| Multimeter | Amazon | $12 |
| LiPo balance charger (B3) | Amazon | $10 |

**Tools subtotal: ~$50** (one-time, shared)

---

## Cost Summary

### 2 Bots (SG90, prototype)

| Category | Cost |
|---|---|
| Electronics × 2 | $150 |
| Filament × 2 | $12 |
| Shared (screws, wires, resistors) | $12 |
| Tools (one-time) | $50 |
| **Total** | **~$225** |

### 4 Bots (SG90, prototype)

| Category | Cost |
|---|---|
| Electronics × 4 | $300 |
| Filament × 4 | $24 |
| Shared (screws, wires, resistors) | $12 |
| Tools (one-time) | $50 |
| **Total** | **~$386** |

### 4 Bots (MG90S, durable)

| Category | Cost |
|---|---|
| Electronics × 4 | $400 |
| Filament × 4 | $24 |
| Shared | $12 |
| Tools | $50 |
| **Total** | **~$486** |

**Recommendation:** Start with 2 bots on SG90s (~$225). Validate walking, then decide on 4 + MG90S upgrade.

---

## Where to Buy

- **Amazon:** Fastest. Search "SG90 servo 12 pack", "PCA9685", "INA219", "ESP32 WROVER", "TSOP38238", "2S 1000mAh LiPo"
- **AliExpress:** Cheapest (~30% less), 2-3 week shipping. Same part names.
- **Adafruit:** Higher quality breakouts (PCA9685, INA219), ~2x price, great docs.
- **MakerWorld:** SIXpack 3D model (free download, account required for some files)

**Tip:** Buy servos in bulk packs (12 or 20). Individual SG90s cost more per unit.

---

## Battery & Power Notes

- **Servo rail:** 5V from LM2596 buck (set to 5.0V). 12 SG90s can pull 3A+ when walking. Do NOT power servos from ESP32 5V pin.
- **ESP32:** Powered from USB during dev, from 5V rail (via VIN) on battery.
- **Low voltage cutoff:** 2S LiPo must not drop below 6.0V (3.0V/cell). The $2 alarm beeps at 6.6V — land and charge.
- **Charging:** B3 compact balance charger (~$10), charges via balance lead in ~1h.

---

## Build Order

1. Print SIXpack chassis + legs (~12h)
2. Assemble servos into legs, mount to chassis
3. Wire PCA9685 → servos (I2C to ESP32)
4. Flash firmware, test servo sweep
5. Add INA219s (one per leg power rail)
6. Add IR receivers in funnel shrouds
7. Mount camera, connect to ESP32
8. Battery + buck converter, power test
9. Walk test → tune gait → connect to brain

Firmware: `firmware/esp32-vnc/esp32-vnc.ino` in this repo.

---

## Classroom Tier (Budget)

Stripped to the core "fly brain controls robot legs" demo. USB powered,
brain on laptop, no autonomy.

| Part | Qty | $ | Source |
|---|---|---|---|
| ESP32 DEVKIT V1 | 1 | $5 | AliExpress |
| PCA9685 PWM driver | 1 | $3 | AliExpress |
| SG90 servos (12-pack) | 12 | $25 | Amazon |
| Micro USB cable | 1 | $2 | — |
| Jumper wires | 1 pack | $3 | Shared |
| 3D printed parts | 1 set | $6 | PETG |

**Per bot: ~$44 | 2 bots: ~$88 | 4 bots: ~$176**

**Upgrade path** (each is a separate lesson):
1. Add camera → visual navigation
2. Add INA219 → obstacle detection (campaniform sensilla!)
3. Add IR → robot-to-robot communication
4. Add battery → untethered autonomy
