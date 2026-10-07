// Go hexapod simulator — validates BANC VNC motor output produces walking.
// Phase 1: kinematic hexapod with tripod gait (validates physics).
// Phase 2: drive from BANC motor neuron firing rates.
// Visualization: top-down + side view, outputs PNG frames → ffmpeg video.
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
)

const (
	// Body dimensions (mm)
	bodyLen = 120.0
	bodyWid = 80.0
	bodyHgt = 30.0

	// Leg dimensions (mm) — SG90-scale
	coxaLen  = 25.0 // hip to knee (horizontal)
	femurLen = 50.0 // knee to ankle
	tibiaLen = 80.0 // ankle to foot

	// Gait
	stepFreq = 2.5  // Hz
	strideLen = 40.0 // mm per step
)

// Leg position indices
const (
	FrontLeft = iota
	MiddleLeft
	RearLeft
	FrontRight
	MiddleRight
	RearRight
)

type Vec3 struct{ X, Y, Z float64 }

type Leg struct {
	// Mount point on body (relative to body center)
	mount Vec3
	// Joint angles (radians)
	hipAngle  float64 // yaw (forward/back swing)
	kneeAngle float64 // pitch (lift)
	// Foot position (world)
	foot Vec3
	// Gait phase (0-1)
	phase float64
	// Is this leg in stance (on ground)?
	stance bool
}

type Hexapod struct {
	pos   Vec3   // body center (world)
	yaw   float64 // heading
	legs  [6]Leg
	tick  int
}

func NewHexapod() *Hexapod {
	h := &Hexapod{pos: Vec3{0, 0, 50}}
	// Leg mounts: front/middle/rear, left/right
	mounts := [6]Vec3{
		{40, 40, 0},   // front left
		{0, 42, 0},    // middle left
		{-40, 40, 0},  // rear left
		{40, -40, 0},  // front right
		{0, -42, 0},   // middle right
		{-40, -40, 0}, // rear right
	}
	for i := range h.legs {
		h.legs[i].mount = mounts[i]
		// Tripod gait: FL+MR+RL phase 0, FR+ML+RR phase 0.5
		if i == FrontLeft || i == MiddleRight || i == RearLeft {
			h.legs[i].phase = 0
		} else {
			h.legs[i].phase = 0.5
		}
	}
	return h
}

// Tripod gait: legs alternate between stance (push back) and swing (lift + move forward)
func (h *Hexapod) step(dt float64) {
	h.tick++
	t := float64(h.tick) * dt

	for i := range h.legs {
		leg := &h.legs[i]
		phase := math.Mod(leg.phase+t*stepFreq, 1.0)

		if phase < 0.5 {
			// Stance: foot on ground, pushing body forward
			leg.stance = true
			// Hip sweeps from front to back
			progress := phase / 0.5 // 0→1
			leg.hipAngle = (0.5 - progress) * 0.6 // ±0.3 rad
			leg.kneeAngle = -0.3 // leg extended down
		} else {
			// Swing: foot lifted, moving forward
			leg.stance = false
			progress := (phase - 0.5) / 0.5
			leg.hipAngle = (progress - 0.5) * 0.6
			// Lift during swing (sinusoidal)
			leg.kneeAngle = -0.3 + math.Sin(progress*math.Pi)*0.5
		}

		// Forward kinematics: mount → hip → knee → foot
		h.updateFoot(i)
	}

	// Body moves forward during stance
	// (simplified: constant velocity when any legs in stance)
	h.pos.X += strideLen * stepFreq * dt * 0.5
}

func (h *Hexapod) updateFoot(i int) {
	leg := &h.legs[i]
	// Hip position (body frame → world)
	hx := h.pos.X + leg.mount.X*math.Cos(h.yaw) - leg.mount.Y*math.Sin(h.yaw)
	hy := h.pos.Y + leg.mount.X*math.Sin(h.yaw) + leg.mount.Y*math.Cos(h.yaw)
	hz := h.pos.Z

	// Coxa (hip yaw)
	cx := hx + coxaLen*math.Cos(leg.hipAngle+h.yaw)
	cy := hy + coxaLen*math.Sin(leg.hipAngle+h.yaw)

	// Femur + tibia (simplified 2-link in vertical plane)
	// Knee angle controls height
	reach := femurLen + tibiaLen*math.Cos(leg.kneeAngle)
	height := tibiaLen * math.Sin(-leg.kneeAngle)

	// Foot position
	fx := cx + reach*math.Cos(leg.hipAngle+h.yaw)*0.5
	fy := cy + reach*math.Sin(leg.hipAngle+h.yaw)*0.5
	fz := hz - 40 + height // 40 = nominal leg length

	if leg.stance {
		fz = 0 // foot on ground
	}

	leg.foot = Vec3{fx, fy, fz}
}

// ── Visualization ──

func (h *Hexapod) renderTopDown() *image.RGBA {
	W, H := 600, 600
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	// Background
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			img.Set(x, y, color.RGBA{20, 20, 20, 255})
		}
	}

	// World to pixel: 1mm = 2px, center at (300,300)
	toPx := func(x, y float64) (int, int) {
		return int(300 + x*2), int(300 - y*2)
	}

	// Body (rotated rect)
	bx, by := toPx(h.pos.X, h.pos.Y)
	// Simple: draw body as circle
	for dy := -20; dy <= 20; dy++ {
		for dx := -30; dx <= 30; dx++ {
			px, py := bx+dx, by+dy
			if px >= 0 && px < W && py >= 0 && py < H {
				// Ellipse check
				if float64(dx*dx)/900+float64(dy*dy)/400 < 1 {
					img.Set(px, py, color.RGBA{100, 100, 100, 255})
				}
			}
		}
	}

	// Legs
	legColors := []color.RGBA{
		{255, 100, 100, 255}, {100, 255, 100, 255}, {100, 100, 255, 255},
		{255, 255, 100, 255}, {255, 100, 255, 255}, {100, 255, 255, 255},
	}
	for i, leg := range h.legs {
		// Mount to foot line
		mx, my := toPx(
			h.pos.X+leg.mount.X*math.Cos(h.yaw)-leg.mount.Y*math.Sin(h.yaw),
			h.pos.Y+leg.mount.X*math.Sin(h.yaw)+leg.mount.Y*math.Cos(h.yaw),
		)
		fx, fy := toPx(leg.foot.X, leg.foot.Y)
		drawLine(img, mx, my, fx, fy, legColors[i])
		// Foot dot (green if stance, red if swing)
		fc := color.RGBA{0, 255, 0, 255}
		if !leg.stance {
			fc = color.RGBA{255, 0, 0, 255}
		}
		for dy := -3; dy <= 3; dy++ {
			for dx := -3; dx <= 3; dx++ {
				if fx+dx >= 0 && fx+dx < W && fy+dy >= 0 && fy+dy < H {
					img.Set(fx+dx, fy+dy, fc)
				}
			}
		}
	}

	// Info text (simple)
	// (skip text rendering for now — keep it minimal)

	return img
}

func drawLine(img *image.RGBA, x0, y0, x1, y1 int, c color.RGBA) {
	dx := abs(x1 - x0)
	dy := abs(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx - dy
	for {
		if x0 >= 0 && x0 < img.Bounds().Dx() && y0 >= 0 && y0 < img.Bounds().Dy() {
			img.Set(x0, y0, c)
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x0 += sx
		}
		if e2 < dx {
			err += dx
			y0 += sy
		}
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func mainKinematic() {
	h := NewHexapod()
	dt := 0.02 // 50Hz

	os.MkdirAll("/tmp/hexapod_frames", 0755)

	// Simulate 10 seconds = 500 frames
	for f := 0; f < 500; f++ {
		h.step(dt)
		img := h.renderTopDown()
		fn := fmt.Sprintf("/tmp/hexapod_frames/frame_%04d.png", f)
		fh, _ := os.Create(fn)
		png.Encode(fh, img)
		fh.Close()
		if f%100 == 0 {
			fmt.Printf("frame %d/500, pos=(%.1f, %.1f)\n", f, h.pos.X, h.pos.Y)
		}
	}
	fmt.Println("done — 500 frames in /tmp/hexapod_frames/")
}
