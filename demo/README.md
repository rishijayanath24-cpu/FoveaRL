# FoveaRL demo

Open **FoveaRL_demo.html** in Chrome or Edge (double-click; fully offline, one 3.5 MB file). Best on a 1080p screen, full-screen (F11).

Built with **Three.js** (WebGL). The demo replays real episodes recorded from the paper's simulator with the five-seed trained DQN agents (agent seed 0) and the baselines, on the first four held-out evaluation seeds of each network profile (10000-10003, not hand-picked).

## What is on screen
- **Arena (left):** the same obstacle course and the same network for six controllers at once. Drag to orbit, wheel to zoom. The ghost robot is where the operator *thinks* the robot is (the delayed frame).
- **Operator view (right):** the actual foveated rendering. Sharp circle = fovea, periphery rendered at lower resolution and bitrate (blockiness). The ring is where the fovea *was centred*, one latency ago; the cross is where the operator looks now. Amber/scanlines + "FRAME FROZEN" = late frame; "LINK OUTAGE" = bandwidth at its floor.
- **Telemetry:** available bandwidth vs sent bitrate, frame age, quality proxy. Red bands = outages.
- **Scoreboard:** progress, collisions, frame age, quality, sent rate, status. Click a row to look through that controller's eyes.
- **Foveation lab:** your mouse is the gaze; choose radius / periphery resolution / bitrate / latency and see the quality proxy and sent rate computed from the paper's equations.

## Keys
Space play/pause · 1-6 pick controller · arrows seek · T tour · Esc close panel

## 3-minute script
1. (20 s) Press **Guided tour**. "A VR operator drives a robot over a network. Same course, same link, six controllers."
2. (40 s) Congested step: "Fixed-High sends everything at 12 Mbps; the link cannot carry it, frames freeze, 0% success over 40 episodes."
3. (40 s) QoE-only agent: "The learned agent shrinks the stream to the link and keeps the sharp region where the operator looks."
4. (30 s) Intermittent step: "A network pattern never seen in training. It matches the best fixed setting on most unseen patterns but not this one: 88% vs 98%. We report that."
5. (30 s) Task-aware step: "We expected a task-aware reward to help. It made things worse. Ablations traced it to the control-error term."
6. (20 s) Results panel, then the **Foveation lab**: let a judge move the mouse.

## Be upfront with judges
It is a simulation replay, not a headset; the quality value is a parametric proxy; operator and network models are assumed (all stated in the paper and on screen).

## Rebuild
`python build_demo.py` bundles `src/` + `data.js` + `vendor_dl/three.min.js`. Regenerate `data.js` with `vr_teleop_rl/export_demo.py`.
