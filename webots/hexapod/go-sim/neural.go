// Neural-driven hexapod: BANC motor neurons → servo angles.
// Obstacles with campaniform sensilla (load) feedback.
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
)

// ── Motor neuron mapping ──
// Maps BANC motor neuron groups to leg servos.
// In Phase 1, we use synthetic firing rates (tripod pattern).
// In Phase 2, these come from the live BANC sim via SensV/Spkc.

type MotorDrive struct {
	// Firing rates (Hz) for each leg's hip and knee motor pools
	// Index: leg (0-5), 0=hip flexor, 1=hip extensor, 2=knee flexor, 3=knee extensor
	rates [6][4]float64
}

// Tripod pattern generator (synthetic — replaces BANC in Phase 1)
func (md *MotorDrive) tripod(t float64, speed float64) {
	for leg := 0; leg < 6; leg++ {
		// Phase: tripod groups
		var phase float64
		if leg == FrontLeft || leg == MiddleRight || leg == RearLeft {
			phase = 0
		} else {
			phase = 0.5
		}
		p := math.Mod(phase+t*stepFreq, 1.0)

		if p < 0.5 {
			// Stance: extensors active (push)
			md.rates[leg][1] = 80 * speed // hip extensor
			md.rates[leg][3] = 60 * speed // knee extensor
			md.rates[leg][0] = 10         // hip flexor (low)
			md.rates[leg][2] = 10         // knee flexor (low)
		} else {
			// Swing: flexors active (lift + swing forward)
			md.rates[leg][0] = 80 * speed
			md.rates[leg][2] = 70 * speed
			md.rates[leg][1] = 10
			md.rates[leg][3] = 10
		}
	}
}

// Convert motor neuron rates to servo angles.
// Flexor/extensor antagonistic pair → joint angle.
func (md *MotorDrive) toServoAngles() [6][2]float64 {
	var angles [6][2]float64
	for leg := 0; leg < 6; leg++ {
		// Hip: flexor - extensor → angle
		hipDiff := md.rates[leg][0] - md.rates[leg][1]
		angles[leg][0] = hipDiff / 80.0 * 0.3 // ±0.3 rad

		// Knee: flexor - extensor → angle
		kneeDiff := md.rates[leg][2] - md.rates[leg][3]
		angles[leg][1] = -0.3 + kneeDiff/70.0*0.4
	}
	return angles
}

// ── Obstacles ──

type Obstacle struct {
	x, y, r float64 // position and radius
}

type World struct {
	hexapod   *Hexapod
	obstacles []Obstacle
	motor     *MotorDrive
	// CS (campaniform sensilla) feedback: load per leg
	csLoad [6]float64
}

func NewWorld() *World {
	w := &World{
		hexapod: NewHexapod(),
		motor:   &MotorDrive{},
		obstacles: []Obstacle{
			{200, 30, 25},
			{350, -40, 30},
			{500, 20, 20},
		},
	}
	return w
}

func (w *World) step(dt float64) {
	t := float64(w.hexapod.tick) * dt

	// 1. Brain decides: tripod gait (Phase 1 synthetic)
	//    Phase 2: replace with BANC DNp09/DNa02 → motor neuron rates
	speed := 1.0
	// Slow down if front legs hit obstacle (CS feedback → brain)
	for leg := 0; leg < 6; leg++ {
		if w.csLoad[leg] > 0.5 && (leg == FrontLeft || leg == FrontRight) {
			speed = 0.3 // cautious
		}
	}
	w.motor.tripod(t, speed)

	// 2. Motor neurons → servo angles
	angles := w.motor.toServoAngles()
	for leg := 0; leg < 6; leg++ {
		w.hexapod.legs[leg].hipAngle = angles[leg][0]
		w.hexapod.legs[leg].kneeAngle = angles[leg][1]
		// Update stance from motor pattern (extensors = stance)
		w.hexapod.legs[leg].stance = w.motor.rates[leg][1] > w.motor.rates[leg][0]
		w.hexapod.updateFoot(leg)
	}

	// 3. Collision detection → CS load sensors
	for leg := 0; leg < 6; leg++ {
		w.csLoad[leg] = 0
		fx, fy := w.hexapod.legs[leg].foot.X, w.hexapod.legs[leg].foot.Y
		for _, ob := range w.obstacles {
			d := math.Hypot(fx-ob.x, fy-ob.y)
			if d < ob.r+10 {
				// Foot near obstacle → load signal
				w.csLoad[leg] = 1.0 - d/(ob.r+10)
			}
		}
	}

	// 4. Move body (only if front is clear)
	blocked := false
	for _, ob := range w.obstacles {
		d := math.Hypot(w.hexapod.pos.X-ob.x, w.hexapod.pos.Y-ob.y)
		if d < ob.r+40 {
			blocked = true
			break
		}
	}
	if !blocked {
		w.hexapod.pos.X += strideLen * stepFreq * dt * 0.5 * speed
	}

	w.hexapod.tick++
}

func (w *World) render() *image.RGBA {
	W, H := 800, 600
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			img.Set(x, y, color.RGBA{20, 20, 20, 255})
		}
	}
	toPx := func(x, y float64) (int, int) {
		return int(100 + x*1.0), int(300 - y*1.0)
	}

	// Obstacles (gray circles)
	for _, ob := range w.obstacles {
		ox, oy := toPx(ob.x, ob.y)
		r := int(ob.r)
		for dy := -r; dy <= r; dy++ {
			for dx := -r; dx <= r; dx++ {
				if dx*dx+dy*dy <= r*r && ox+dx >= 0 && ox+dx < W && oy+dy >= 0 && oy+dy < H {
					img.Set(ox+dx, oy+dy, color.RGBA{80, 80, 80, 255})
				}
			}
		}
	}

	// Body
	h := w.hexapod
	bx, by := toPx(h.pos.X, h.pos.Y)
	for dy := -15; dy <= 15; dy++ {
		for dx := -25; dx <= 25; dx++ {
			if bx+dx >= 0 && bx+dx < W && by+dy >= 0 && by+dy < H {
				if float64(dx*dx)/625+float64(dy*dy)/225 < 1 {
					img.Set(bx+dx, by+dy, color.RGBA{120, 120, 120, 255})
				}
			}
		}
	}

	// Legs with CS load indication
	legColors := []color.RGBA{
		{255, 100, 100, 255}, {100, 255, 100, 255}, {100, 100, 255, 255},
		{255, 255, 100, 255}, {255, 100, 255, 255}, {100, 255, 255, 255},
	}
	for i, leg := range h.legs {
		mx, my := toPx(
			h.pos.X+leg.mount.X*math.Cos(h.yaw)-leg.mount.Y*math.Sin(h.yaw),
			h.pos.Y+leg.mount.X*math.Sin(h.yaw)+leg.mount.Y*math.Cos(h.yaw),
		)
		fx, fy := toPx(leg.foot.X, leg.foot.Y)
		// Leg color shifts to yellow if CS loaded (hitting obstacle)
		c := legColors[i]
		if w.csLoad[i] > 0.3 {
			c = color.RGBA{255, 255, 0, 255} // yellow = contact!
		}
		drawLine(img, mx, my, fx, fy, c)
		// Foot
		fc := color.RGBA{0, 255, 0, 255}
		if !leg.stance {
			fc = color.RGBA{255, 0, 0, 255}
		}
		for dy := -2; dy <= 2; dy++ {
			for dx := -2; dx <= 2; dx++ {
				if fx+dx >= 0 && fx+dx < W && fy+dy >= 0 && fy+dy < H {
					img.Set(fx+dx, fy+dy, fc)
				}
			}
		}
	}

	return img
}

func mainNeural() {
	w := NewWorld()
	dt := 0.02

	os.MkdirAll("/tmp/hexapod_neural", 0755)

	for f := 0; f < 600; f++ {
		w.step(dt)
		img := w.render()
		fn := fmt.Sprintf("/tmp/hexapod_neural/frame_%04d.png", f)
		fh, _ := os.Create(fn)
		png.Encode(fh, img)
		fh.Close()
		if f%100 == 0 {
			fmt.Printf("frame %d/600 pos=(%.0f,%.0f) cs=%v\n",
				f, w.hexapod.pos.X, w.hexapod.pos.Y, w.csLoad)
		}
	}
	fmt.Println("done")
}
