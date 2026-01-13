# Risiko AlphaZero Starter

This repository is a clean restart of the Risiko project, oriented around an AlphaZero-style learning agent.
The focus is to keep a **simple, well-structured core** that can already self-play, learn, and improve.

## Goals

- ✅ Start with a minimal, playable Risk-like environment.
- ✅ Provide AlphaZero-style MCTS + neural network scaffolding.
- ✅ Enable self-play data generation and training from scratch.
- ✅ Keep the code modular and easy to extend.

## Project structure

```
.
├── legacy/                 # Previous prototype code kept for reference
├── scripts/                # Entry points for self-play and training
├── src/
│   └── risiko/
│       ├── az/             # AlphaZero components (network, MCTS, training)
│       ├── game/           # Risk-like game rules + environment
│       └── utils/          # Helpers and serialization
└── README.md
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run self-play to generate data:

```bash
python scripts/run_self_play.py --games 8 --output data/self_play.jsonl --mcts-sims 128 \
  --temperature 1.0 --temperature-decay 0.97 --min-temperature 0.25
```

Control which players act randomly during self-play:

```bash
# Use random players 0 and 1 (0-based indices), or "all".
python scripts/run_self_play.py --random-players 0,1
```

Run the live GUI viewer (random policy for now):

```bash
pip install -e ".[gui]"
python scripts/run_gui.py --delay 0.8
```

Use AI vs random (MCTS + policy/value network) in the GUI:

```bash
python scripts/run_gui.py --players ai,random --delay 0.6
```

Load a trained model for AI players:

```bash
python scripts/run_gui.py --players ai,random --model data/model.pt
```

Tune GUI runtime options:

```bash
python scripts/run_gui.py --auto-reset --seed 7 --mcts-sims 128 --mcts-cpuct 1.2
```

Hotkeys while the GUI is running:

- **Space**: pause/resume
- **N**: step one move (while paused)
- **R**: reset the game

> **Windows note:** Python 3.13 users will get `pygame-ce` automatically because `pygame` does not publish official
> wheels for 3.13 yet. If you prefer `pygame`, use Python 3.12 or install the required build tools per
> https://www.pygame.org/wiki/CompileWindows.

Train a small model on the generated data:

```bash
python scripts/train.py --data data/self_play.jsonl --epochs 4 --batch-size 64 \
  --learning-rate 0.002 --output data/model.pt
```

Run an iterative self-play training loop with a replay buffer (recommended for gradual gains):

```bash
python scripts/run_training_loop.py --iterations 6 --games-per-iteration 8 \
  --buffer-size 5000 --mcts-sims 128 --temperature 1.0 --temperature-decay 0.97 \
  --min-temperature 0.25 --output-dir data/training
```

## Design notes

- The current game rules are intentionally simplified to make the learning loop tight and fast.
- The `ActionSpace` is fixed and indexed so the policy head can predict logits for every action.
- The MCTS uses PUCT with Dirichlet noise for exploration.

## Next steps

- Expand the map to the full Risk board.
- Replace the simplified combat resolution with standard dice rules.
- Add evaluation, Elo tracking, and checkpointing.

---

If you want to extend or refactor, start in `src/risiko/game` and `src/risiko/az`.
