# VR teleoperation foveated-streaming RL simulation

Code behind the FoveaRL study (paper figures and LaTeX tables in `../paper/`). Pure Python (NumPy, PyTorch CPU, SciPy, Matplotlib); no ROS/Gazebo/headset.

| File | Purpose |
|---|---|
| `sim.py` | Simulator: network profiles (4 main + 5 held-out), foveated streaming + quality proxy, gaze, operator, robot, reward. `DEFAULT_CFG` holds the constants the sensitivity study varies |
| `agents.py` | Fixed and rule-based baselines, double DQN training |
| `train_all.py` | Tunes the rule baseline, trains 5 seeds x {task-aware, QoE-only}: `python train_all.py 700` and `python train_all.py 2000 _r2` |
| `train_extra.py` | Reward ablations (control-error only, collision only, action persistence; 2000 ep.) and the 6 sensitivity settings (700 ep., 3 seeds) |
| `evaluate.py` | Original held-out evaluation, Wilcoxon summary, figures 1, 3, 4: `python evaluate.py _r2` |
| `evaluate2.py` | Extended evaluation: ablations, 5 extra unseen network patterns, sensitivity settings, reward components |
| `analyze.py` | Two-level bootstrap (agents x paired episodes), Holm correction, tables, forest plot |
| `make_tables.py` | LaTeX tables for the main and training-budget results |
| `regress.py` | Checks the extended simulator reproduces the stored main results exactly |
| `sanity.py`, `show.py`, `show_stats.py`, `peek.py` | Quick inspection helpers |

Reproduce:

```
python train_all.py 700 ; python train_all.py 2000 _r2 ; python train_extra.py
python evaluate.py ; python evaluate.py _r2 ; python evaluate2.py all
python analyze.py ; python make_tables.py
```

Training: about 4 min (700 ep.) and 12-16 min (2000 ep.) per batch on 16 cores; `train_extra.py` about 45 min. Evaluation is seeded and deterministic.

Caveats (also in the paper): quality is a parametric proxy, not SSIM/VMAF; operator, gaze and obstacle-perception models are assumptions; networks are synthetic.
