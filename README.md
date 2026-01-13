# Taxi Cab Problem — Reinforcement Learning

A Q-learning agent that learns to navigate a taxi in a 5×5 grid world, pick up passengers, and drop them at their destination.

**[▶ Live Demo](https://rednayan.github.io/rl-taxi-cab-problem/)**

![Taxi Environment](https://gymnasium.farama.org/_images/taxi.gif)

## The Problem

Based on the classic [Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment. The taxi must:

1. Navigate a 5×5 grid with walls
2. Pick up a passenger from one of 4 locations (R, G, Y, B)
3. Drop them off at another location
4. Do this as efficiently as possible

**Rewards:**
- +20 for successful drop-off
- -1 for each step (encourages efficiency)
- -10 for illegal pickup/drop-off attempts

**State space:** 500 states (5×5 grid × 5 passenger locations × 4 destinations)

**Actions:** 6 (Up, Down, Left, Right, Pickup, Drop)

## The Solution

This project implements **Q-learning**, a model-free reinforcement learning algorithm. The agent learns a Q-table mapping state-action pairs to expected rewards through trial and error.

```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

**Hyperparameters:**
- Learning rate (α): 0.1
- Discount factor (γ): 0.99
- Exploration rate (ε): 0.1
- Episodes: 5000

After training, the agent learns the optimal policy and consistently solves the task in minimal steps.

## Project Structure

```
├── main.c          # Training loop, visualization, Q-learning
├── taxi.c          # Environment logic (step, reset, rewards)
├── taxi.h          # Environment struct and function declarations
```

## Building Locally

Requires [raylib](https://www.raylib.com/) for visualization.

- **Linux:** raylib 5.5 is included in the repo — no installation needed.

- **For other systems:** download the source code of [raylib](https://github.com/raysan5/raylib.git) 

```bash
# manually with gcc
gcc -o taxi main.c taxi.c -lraylib -lGL -lm -lpthread

# or Using the nob build system
# update the `nob.c` file with appropiate file path for a `nob` build.
./nob


```

## License

MIT
