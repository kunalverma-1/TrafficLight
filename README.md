# Intelligent Traffic Light Management System with Ambulance Prioritization

## Project Overview
This project is an AI-powered Intelligent Traffic Light Management System designed to optimize urban traffic flow using Reinforcement Learning (RL) and the SUMO (Simulation of Urban MObility) traffic simulator. The system dynamically controls traffic lights at intersections to minimize vehicle waiting times and congestion, while also prioritizing ambulances to ensure their rapid and safe passage through the city network.

## Key Features
- **AI-Driven Traffic Light Control:** Utilizes RL agents to learn optimal traffic light phase switching strategies for each junction, adapting to real-time traffic conditions.
- **Ambulance Prioritization:** Special logic is implemented to detect ambulances and prioritize their movement through intersections, including penalizing ambulance waiting time in the RL reward function.
- **Multi-Junction Support:** The system can handle multiple intersections, with per-junction RL models for scalable deployment.
- **SUMO Integration:** Leverages SUMO for realistic traffic simulation, including custom city networks and route generation.
- **Command-Line Interface:** Flexible training and testing modes via command-line arguments.
- **Robust Error Handling:** Includes error handling for SUMO connection issues and model file mismatches.

## Technical Stack
- **Programming Language:** Python 3.x
- **Simulation Engine:** [SUMO](https://www.eclipse.org/sumo/) (Simulation of Urban MObility)
- **AI/ML Frameworks:**
  - NumPy (for numerical operations)
  - PyTorch or Keras/TensorFlow (for RL agent implementation, as inferred from typical RL workflows)
- **Visualization:** Matplotlib (for plotting training results)
- **DevOps & Environment:**
  - Virtual environments (venv)
  - Requirements management via `requirements.txt`
  - Command-line tools for training, testing, and route generation

## Project Structure
- `train2.py`: Main script for training/testing RL agents, handling SUMO integration, ambulance logic, and per-junction model management.
- `configuration.sumocfg`: SUMO configuration file specifying the network and route files.
- `maps/`: Contains SUMO network (`.net.xml`) and route (`.rou.xml`) files for different city scenarios, as well as route generation scripts.
- `models/`: Stores trained RL models for each junction.
- `plots/`: Output directory for training and evaluation plots.
- `requirements.txt`: Lists Python dependencies.

## Reinforcement Learning Approach
- **State Representation:** Each RL agent observes the current state of its controlled junction, including traffic light phases, vehicle counts, and ambulance presence.
- **Action Space:** The agent selects the next traffic light phase for its junction.
- **Reward Function:** Designed to minimize overall vehicle waiting time, with additional penalties for ambulance delays to ensure their prioritization.
- **Training:** Agents are trained using episodes simulated in SUMO, with per-junction models saved for modularity and scalability.

## Ambulance Logic
- **Detection:** Ambulances are identified in the SUMO simulation via vehicle type or route file logic.
- **Prioritization:** When an ambulance is detected approaching or waiting at a junction, the RL agent is incentivized (via the reward function) to switch phases to allow the ambulance to pass with minimal delay.
- **Route Generation:** Custom route files (`city3.rou.xml`, etc.) ensure ambulances spawn at realistic intervals and traverse the network from boundary edges to endpoints.

## Skills & Technologies Demonstrated
- Reinforcement Learning (RL) for control systems
- Traffic simulation and modeling with SUMO
- Python programming and scripting
- AI/ML model management and evaluation
- Command-line interface development
- Data visualization with Matplotlib
- Error handling and robust system design
- DevOps basics: virtual environments, dependency management

## How to Run
1. **Set up the Python environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Train the RL models:**
   ```cmd
   python train2.py --mode train --city city3
   ```
3. **Test or run the simulation:**
   ```cmd
   python train2.py --mode test --city city3
   ```
4. **Generate new routes (if needed):**
   ```cmd
   python maps\randomTrips.py -n maps\city3.net.xml -o maps\city3.rou.xml --additional-options
   ```

## DevOps & Best Practices
- Use virtual environments to isolate dependencies.
- Store models and outputs in dedicated directories (`models/`, `plots/`).
- Modular code structure for easy extension to new cities or junctions.
- Clear error messages and robust handling of simulation or model issues.

## Acknowledgements
- [SUMO](https://www.eclipse.org/sumo/) for traffic simulation.
- Open-source Python libraries for AI and data science.

---
For further details, see the code in `train2.py` and the SUMO configuration and route files in the `maps/` directory.

## Recent Enhancements
### Emergency Vehicle Prioritization Framework
- The original ambulance-only prioritization mechanism was extended into a multi-class emergency response system.

- Priority Levels:

- Ambulance (Priority 3)
- Fire Truck (Priority 2)
- Police Vehicle (Priority 1)

- The reinforcement learning reward function was modified to dynamically adjust penalties based on emergency vehicle priority, enabling more adaptive traffic signal control.

### Traffic Analytics Module
- An analytics layer was introduced to evaluate traffic performance through:
- Congestion Score
- Traffic Efficiency Score
- Emergency Priority Score

The analytics framework supports future visualization and benchmarking extensions.