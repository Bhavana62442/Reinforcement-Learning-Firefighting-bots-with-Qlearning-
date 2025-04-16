# Firefighting Simulation (Q-learning + MAPF + Chain)

This is a Python-based simulation of a multi-agent firefighting scenario using Q-learning for decision-making, A* pathfinding (MAPF - Multi-Agent Path Finding), and a chain-based water-passing mechanism. Agents navigate a 10x10 grid to extinguish fires, refill water, and collaborate by passing water smong agents.

## Features
- **Multi-Agent System**: Multiple agents (up to 6 in this setup) work to extinguish fires.
- **Q-learning**: Agents learn optimal actions (move to fire, move to water, pass water) using a Q-table.
- **A* Pathfinding**: Efficient navigation around obstacles and occupied cells.
- **Chain Mechanism**: Agents can pass water to adjacent dehydrated agents.
- **Visualization**: Real-time grid display using Pygame with animated fire GIFs and static icons.
- **Fire Spread**: Fires can spread to adjacent empty cells with a probability.

## Requirements

### Python Libraries
Install the required dependencies using the provided `requirements.txt`:
- `numpy>=1.19`
- `pygame>=2.0`
- `Pillow>=8.0`

To install:
```bash
pip install -r requirements.txt