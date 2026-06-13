from __future__ import absolute_import
from __future__ import print_function


import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

import os
import sys
import time
import optparse
import random
import serial
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# we need to import python modules from the $SUMO_HOME/tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

# HARD-CODED LANE AND PHASE INFO FOR CITY3 NETWORK
# This code is now specific to city3.net.xml and city3.rou.xml
# Update these lists if your city3 network changes!
CITY3_JUNCTIONS = ["gneJ15", "gneJ4", "gneJ9"]
CITY3_LANES = {
    "gneJ15": ["gneE19_0", "gneE19_1", "gneE20_0", "gneE20_1", "gneE21_0", "gneE21_1", "gneE22_0", "gneE22_1"],
    "gneJ4":  ["gneE7_0", "gneE7_1", "gneE8_0", "gneE8_1", "gneE9_0", "gneE9_1", "gneE10_0", "gneE10_1"],
    "gneJ9":  ["gneE15_0", "gneE15_1", "gneE16_0", "gneE16_1", "gneE17_0", "gneE17_1", "gneE18_0", "gneE18_1"]
}
CITY3_PHASES = {
    "gneJ15": 8,  # update if your network has a different number of phases
    "gneJ4": 8,
    "gneJ9": 8
}

def get_vehicle_numbers(lanes):
    vehicle_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = 0
        for k in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(k) > 10:
                vehicle_per_lane[l] += 1
    return vehicle_per_lane


def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time


def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)


def get_emergency_vehicle_presence(lanes):
    """
    Returns:
    3 -> ambulance present
    2 -> fire truck present
    1 -> police vehicle present
    0 -> no emergency vehicle
    """

    emergency_per_lane = []

    for l in lanes:

        priority = 0

        for vid in traci.lane.getLastStepVehicleIDs(l):

            vehicle_type = traci.vehicle.getTypeID(vid)

            if vehicle_type == "ambulance":
                priority = 3

            elif vehicle_type == "firetruck":
                priority = max(priority, 2)

            elif vehicle_type == "police":
                priority = max(priority, 1)

        emergency_per_lane.append(priority)

    return emergency_per_lane


def filter_known_lanes(lane_list):
    known = set(traci.lane.getIDList())
    return [l for l in lane_list if l in known]


class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.linear1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.linear2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.linear3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims_per_junction,  # dict: junction_number -> input_dims
        fc1_dims,
        fc2_dims,
        batch_size,
        n_actions_per_junction,   # dict: junction_number -> n_actions
        junctions,
        max_memory_size=100000,
        epsilon_dec=5e-4,
        epsilon_end=0.05,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100
        self.Q_eval = {}
        self.memory = {}
        self.n_actions_per_junction = n_actions_per_junction
        self.input_dims_per_junction = input_dims_per_junction
        for junction in junctions:
            input_dims = input_dims_per_junction[junction]
            n_actions = n_actions_per_junction[junction]
            self.Q_eval[junction] = Model(
                self.lr, input_dims, self.fc1_dims, self.fc2_dims, n_actions
            )
            self.memory[junction] = {
                "state_memory": np.zeros((self.max_mem, input_dims), dtype=np.float32),
                "new_state_memory": np.zeros((self.max_mem, input_dims), dtype=np.float32),
                "reward_memory": np.zeros(self.max_mem, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem, dtype=np.bool_),
                "mem_cntr": 0,
                "iter_cntr": 0,
            }
    def store_transition(self, state, state_, action, reward, done, junction):
        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = state_
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1
    def choose_action(self, observation, junction):
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval[junction].device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval[junction].forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(range(self.n_actions_per_junction[junction]))
        return action
    def learn(self, junction):
        self.Q_eval[junction].optimizer.zero_grad()
        batch = np.arange(self.memory[junction]['mem_cntr'], dtype=np.int32)
        state_batch = torch.tensor(self.memory[junction]["state_memory"][batch]).to(self.Q_eval[junction].device)
        new_state_batch = torch.tensor(self.memory[junction]["new_state_memory"][batch]).to(self.Q_eval[junction].device)
        reward_batch = torch.tensor(self.memory[junction]['reward_memory'][batch]).to(self.Q_eval[junction].device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval[junction].device)
        action_batch = self.memory[junction]["action_memory"][batch]
        q_eval = self.Q_eval[junction].forward(state_batch)[batch, action_batch]
        q_next = self.Q_eval[junction].forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval[junction].loss(q_target, q_eval).to(self.Q_eval[junction].device)
        loss.backward()
        self.Q_eval[junction].optimizer.step()
        self.iter_cntr += 1
        self.epsilon = (
            self.epsilon - self.epsilon_dec
            if self.epsilon > self.epsilon_end
            else self.epsilon_end
        )
    def save(self, model_name):
        import os
        os.makedirs('models', exist_ok=True)
        for junction in self.Q_eval:
            torch.save(self.Q_eval[junction].state_dict(), f'models/{model_name}_{junction}.bin')
    def load(self, model_name):
        for junction in self.Q_eval:
            path = f'models/{model_name}_{junction}.bin'
            if os.path.exists(path):
                self.Q_eval[junction].load_state_dict(torch.load(path, map_location=self.Q_eval[junction].device))
            else:
                print(f"[WARNING] Model file {path} not found for junction {junction}.")


def run(train=True,model_name="model",epochs=50,steps=500,ard=False):
    if ard:
        arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
        def write_read(x):
            arduino.write(bytes(x, 'utf-8'))
            time.sleep(0.05)
            data = arduino.readline()
            return data
    """Production-ready TraCI control loop for multi-junction, multi-phase, ambulance-prioritizing RL"""
    best_time = np.inf
    total_time_list = list()
    traci.start([
        checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"
    ])
    # Instead of dynamic detection, use hardcoded values for city3
    all_junctions = CITY3_JUNCTIONS
    junctions = all_junctions
    # Filter hardcoded lanes to only those known to SUMO
    junction_lanes = {j: filter_known_lanes(CITY3_LANES[j]) for j in all_junctions}
    input_dims_per_junction = {j: len(junction_lanes[j]) for j in all_junctions}
    n_actions_per_junction = {j: CITY3_PHASES[j] for j in all_junctions}
    # For phases, still get from traci (for phase state strings)
    junction_phases = {j: traci.trafficlight.getCompleteRedYellowGreenDefinition(j)[0].phases for j in all_junctions}

    # For each junction, set n_actions to the number of phases (as in train.py)
    n_actions_per_junction = {j: len(junction_phases[j]) for j in all_junctions}

    brain = Agent(
        gamma=0.99,
        epsilon=0.0,
        lr=0.1,
        input_dims_per_junction=input_dims_per_junction,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=1024,
        n_actions_per_junction=n_actions_per_junction,
        junctions=junctions,  # Use junction IDs
    )

    if not train:
        try:
            brain.load(model_name)
        except RuntimeError as e:
            print(f"\n[ERROR] Failed to load model state dict: {e}\n")
            print("This usually means the saved model was trained with different input dimensions or architecture.\n" 
                  "Please retrain the model or use a compatible model file. Exiting.")
            traci.close()
            sys.exit(1)

    # Print device info for each junction's model
    for junction in brain.Q_eval:
        print(f"Junction {junction} model device: {brain.Q_eval[junction].device}")
    traci.close()
    for e in range(epochs):
        if train:
            traci.start([
                checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"
            ])
        else:
            traci.start([
                checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"
            ])

        # Use hardcoded lane/phase info for city3 in the epoch loop as well
        all_junctions = CITY3_JUNCTIONS
        junctions = all_junctions
        junction_lanes = {j: filter_known_lanes(CITY3_LANES[j]) for j in all_junctions}
        input_dims_per_junction = {j: len(junction_lanes[j]) for j in all_junctions}
        n_actions_per_junction = {j: CITY3_PHASES[j] for j in all_junctions}
        junction_phases = {j: traci.trafficlight.getCompleteRedYellowGreenDefinition(j)[0].phases for j in all_junctions}

        brain = Agent(
            gamma=0.99,
            epsilon=0.0,
            lr=0.1,
            input_dims_per_junction=input_dims_per_junction,
            fc1_dims=256,
            fc2_dims=256,
            batch_size=1024,
            n_actions_per_junction=n_actions_per_junction,
            junctions=junctions,  # Use junction IDs
        )

        # Print device info for each junction's model
        for junction in brain.Q_eval:
            print(f"Junction {junction} model device: {brain.Q_eval[junction].device}")

        print(f"epoch: {e}")
        step = 0
        total_time = 0
        min_duration = 5
        traffic_lights_time = dict()
        prev_wait_time = dict()
        prev_vehicles_per_lane = dict()
        prev_action = dict()
        # Remove the following block to avoid overwriting hardcoded lane/phase info:
        # junction_lanes = {}
        # junction_phases = {}
        # for junction in all_junctions:
        #     junction_lanes[junction] = traci.trafficlight.getControlledLanes(junction)
        #     junction_phases[junction] = traci.trafficlight.getCompleteRedYellowGreenDefinition(junction)[0].phases
        for junction in all_junctions:
            prev_wait_time[junction] = 0
            prev_action[junction] = 0
            traffic_lights_time[junction] = 0
            prev_vehicles_per_lane[junction] = [0] * input_dims_per_junction[junction]

        while step <= steps:
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError as e:
                print("[INFO] SUMO closed the connection (likely all vehicles are gone or a route is invalid). Exiting simulation loop.")
                break
            for junction in all_junctions:
                lanes = junction_lanes[junction]
                phases = junction_phases[junction]
                waiting_time = get_waiting_time(lanes)
                total_time += waiting_time
                emergency_presence = get_emergency_vehicle_presence(lanes)
                # Emergency vehicle prioritization: if any ambulance present, prioritize that phase
                prioritized = False
                for lane_idx, priority in enumerate(emergency_presence):
                    if priority > 0:
                        # Find which phase(s) serve this lane
                        for phase_idx, phase in enumerate(phases):
                            # If this phase gives green to this lane
                            if phase.state[lane_idx] == 'G' or phase.state[lane_idx] == 'g':
                                current_phase = traci.trafficlight.getPhase(junction)
                                if current_phase != phase_idx:
                                    phaseDuration(junction, 6, phase.state)
                                    phaseDuration(junction, min_duration + 10, phase.state)
                                    traffic_lights_time[junction] = min_duration + 10
                                    prev_action[junction] = phase_idx
                                reward = -1 * waiting_time - (priority * 500)  # Strong penalty for ambulance waiting
                                prioritized = True
                                break
                        if not prioritized:
                            print(f"[WARNING] Emergency vehicle detected in lane {lanes[lane_idx]} at junction {junction}, but no phase found to prioritize it.")
                        break
                if prioritized:
                    continue  # Skip RL for this junction this step
                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(lanes)
                    reward = -1 * waiting_time
                    state_ = list(vehicles_per_lane.values())
                    state = prev_vehicles_per_lane[junction]
                    prev_vehicles_per_lane[junction] = state_
                    brain.store_transition(state, state_, prev_action[junction],reward,(step==steps),junction)
                    # Choose action (phase) using RL
                    lane = brain.choose_action(state_, junction)
                    prev_action[junction] = lane
                    # Set the phase for this junction
                    if lane < len(phases):
                        phaseDuration(junction, 6, phases[lane].state)
                        phaseDuration(junction, min_duration + 10, phases[lane].state)
                        traffic_lights_time[junction] = min_duration + 10
                    else:
                        print(f"[WARNING] RL chose phase {lane} for junction {junction}, but only {len(phases)} phases exist.")
                    if ard:
                        ph = str(traci.trafficlight.getPhase("0"))
                        value = write_read(ph)
                    if train:
                        brain.learn(junction)
                else:
                    traffic_lights_time[junction] -= 1
            step += 1
        print("total_time",total_time)
        total_time_list.append(total_time)
        if total_time < best_time:
            best_time = total_time
            if train:
                brain.save(model_name)
        traci.close()
        sys.stdout.flush()
        if not train:
            break
    if train:
        plt.plot(list(range(len(total_time_list))),total_time_list)
        plt.xlabel("epochs")
        plt.ylabel("total time")
        plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
        plt.show()

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="model",
        help="name of model",
    )
    optParser.add_option(
        "--train",
        action = 'store_true',
        default=False,
        help="training or testing",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=50,
        help="Number of epochs",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=500,
        help="Number of steps",
    )
    optParser.add_option(
       "--ard",
        action='store_true',
        default=False,
        help="Connect Arduino", 
    )
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    model_name = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    ard = options.ard
    run(train=train,model_name=model_name,epochs=epochs,steps=steps,ard=ard)
