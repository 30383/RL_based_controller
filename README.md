# DRL-based Controller Design for Buck-Boost Converter

This project explores the use of Deep Reinforcement Learning (DRL) to design and evaluate controllers for a Buck-Boost Converter. The Buck-Boost Converter is a DC-DC converter used in various applications like load converters for electronic devices, electric vehicles, and photovoltaic cells.

## Overview

- **Objective:** To control a Buck-Boost Converter efficiently using DRL algorithms and compare their performance.
- **Algorithms Implemented:**
  - Actor-Critic (AC)
  - Proximal Policy Optimization (PPO)
  - Deep Deterministic Policy Gradient (DDPG)
  - Hybrid model combining DRL with PID control (AC and DDPG)

## Key Concepts

- **Deep Reinforcement Learning (DRL):** Applied to optimize the control strategy of the Buck-Boost Converter.
- **Actor-Critic Architecture:** Used to improve the policy based on real-time feedback.
- **PPO & DDPG:** Advanced RL algorithms tailored for continuous action spaces in power electronic systems.

## Steps to Recreate the Results

1. **Setup the Environment:**
   - Ensure you have MATLAB and Simulink installed (R2023a or later recommended).
   - Clone or download the repository containing the MATLAB scripts and Simulink models.

2. **Folder Structure:**
   - Place all MATLAB files (`.m` files) and Simulink models (`.slx` files) in the same directory.

3. **Running the Simulations:**
   - Open MATLAB and navigate to the folder containing the project files.
   - Run the desired MATLAB script (e.g., `run_drl_simulation.m`). Ensure that the corresponding Simulink model is in the same folder so that it can be accessed during the simulation.
   - The scripts will automatically load the Simulink models and execute the simulations.

4. **Analyzing Results:**
   - Upon completion, the results will be displayed in MATLAB, showing the performance metrics of the different DRL algorithms.
   - You can modify the parameters in the scripts to test different scenarios and compare results.

## Results

- Comparative analysis of different RL-based control strategies showed improved efficiency and stability over traditional methods.

## Acknowledgments

This project was supervised by Dr. Sudha Radhika as part of the Power Electronics course at BITS Pilani, Hyderabad Campus.

## License

This project is licensed under the MIT License.
