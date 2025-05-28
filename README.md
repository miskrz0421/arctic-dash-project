# ArcticDashEnv: Q-learning with Curriculum Learning

A custom Gymnasium environment simulating movement on a frozen lake to collect treasure and return to the starting point. This project implements and tests a Q-learning agent, enhanced with a Curriculum Learning strategy, to effectively navigate and solve the environment. It features a Pygame-based graphical interface for real-time visualization of the agent's actions and learning process.

## Features

*   **Custom Gymnasium Environment:** `ArcticDashEnv` is a custom-built environment extending Gymnasium's API, complete with a defined observation space, action space, dynamics, and reward system.
*   **Q-learning Agent:** A robust Q-learning algorithm is implemented for the agent to learn optimal policies.
*   **Curriculum Learning:** The agent is trained across multiple stages with varying maximum jump limits (`max_jumps`), allowing for observation of adaptation to resource constraints and potential knowledge transfer between stages.
*   **Pygame Graphical Interface:**
    *   Real-time visualization of the game board, agent's position, and dynamic ice states.
    *   Interactive mode with path tracking and an information panel (jumps left, treasure status, FPS, camera modes).
    *   User control over zoom, camera follow, FPS, and full-screen mode.
*   **Factored State Representation:** The agent's state is composed of its spatial position, remaining jumps, and treasure possession status, mapped to a unique index for the Q-table.
*   **Dynamic Hyperparameter Scheduling:** Epsilon-greedy exploration and learning rate are dynamically adjusted throughout training phases to balance exploration and exploitation effectively.
*   **Logging and Plotting:** Detailed logging of training progress and automatic generation of performance plots (rewards, epsilon, learning rate over episodes).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.8+
*   pip (Python package installer)
*   git (for cloning the repository)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/miskrz0421/arctic-dash-project.git
    cd ArcticDashEnv
    ``

2.  **Install the required Python packages:**

    It's recommended to use a virtual environment.

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
    
    Then install the packages:

    ```bash
    pip install gymnasium pygame numpy matplotlib pillow
    ```

## Usage

The project is designed to be run within a Jupyter Notebook environment (e.g., Jupyter Lab, Jupyter Notebook, or Google Colab).

### Running the Environment in Demonstration/Evaluation Mode

This mode allows you to load a pre-trained agent and visualize its performance.

1.  **Open the Notebook:**
    Launch Jupyter Lab/Notebook and open the `ArcticDash_sprawozdanie.ipynb` notebook.

2.  **Execute Setup Cells:**
    Run all cells from Section 0. `Wstęp i Konfiguracja Środowiska Colab/Jupyter` up to the configuration block in Section 5.1. `Ustawienia Eksperymentu`.

3.  **Configure for Demonstration:**
    In **Cell 5 (Configuration)**, ensure the following parameters are set:
    ```python
    PERFORM_TRAINING_OVERALL = False
    RENDER_TRAINING_STAGES = False
    RENDER_EVALUATION_AFTER_EACH_STAGE = True
    NUM_EVALUATION_EPISODES_PER_STAGE = 1 # 1 is sufficient for interactive mode
    ```
    *   **Adjust Jump Limits (Optional):** You can change `CURRICULUM_START_JUMPS` and `CURRICULUM_END_JUMPS` (e.g., from 6 down to 1) to observe agents trained with different jump limits.
        *   **Important:** For evaluation, the corresponding pre-trained Q-table (`q_table_mjX.pkl`) for the chosen `MAX_JUMPS` value must exist in the `q_learning_results_curriculum_e5/map_e5_mjX/` directory.

4.  **Run the Demonstration:**
    Execute **Cell 6 (`Uruchomienie Procesu Curriculum Learning / Demonstracji`)**.
    A Pygame window will open, displaying the agent's performance.

5.  **Interactive Mode Controls (in Pygame window):**
    *   **`R` key:** Reset the current episode (only in interactive mode).
    *   **`C` key:** Toggle camera follow mode.
    *   **`+` / `-` keys:** Adjust zoom level.
    *   **`PgUp` / `PgDn` keys:** Adjust frames per second (FPS).
    *   **`F11` key:** Toggle full-screen mode.
    *   **Close window:** Exit the game.

### Running the Environment in Training Mode

This mode allows you to train a new Q-learning agent from scratch or continue training.

1.  **Open the Notebook:**
    Launch Jupyter Lab/Notebook and open the `ArcticDash_sprawozdanie.ipynb` notebook.

2.  **Execute Setup Cells:**
    Run all cells from Section 0. `Wstęp i Konfiguracja Środowiska Colab/Jupyter` up to the configuration block in Section 5.1. `Ustawienia Eksperymentu`.

3.  **Configure for Training:**
    In **Cell 5 (Configuration)**, set:
    ```python
    PERFORM_TRAINING_OVERALL = True
    RENDER_TRAINING_STAGES = False # Set to True to visualize training (may slow down significantly)
    RENDER_EVALUATION_AFTER_EACH_STAGE = True # To see a demo after training
    ```
    *   **Training Episodes:** Adjust `EPISODES_PER_STAGE` (e.g., `50_000_000` or `100_000_000`) based on desired training time and performance. Be aware that training can be **very time-consuming**, potentially taking many hours or even days depending on the number of episodes and hardware.
    *   **Preload Q-table:** To start training from a specific point or transfer knowledge, set `PRELOAD_QTABLE_PATH` and `PRELOAD_QTABLE_JUMPS` in Cell 5. To start from scratch, set `PRELOAD_QTABLE_PATH = None`.

4.  **Start Training:**
    Execute **Cell 6 (`Uruchomienie Procesu Curriculum Learning / Demonstracji`)**.
    Training progress will be logged to the console and to `.log` files in the `q_learning_results_curriculum_e5` directory. Trained Q-tables (`.pkl` files) and plots will also be saved there.

## Environment Details (ArcticDashEnv)

*   **Grid-based Frozen Lake:** The agent navigates a grid where different cell types have distinct properties.
*   **Goal:** The primary objective is to collect a treasure ('G') and safely return to the starting position ('S').
*   **Hazards:** The lake contains 'Frozen' (F) ice that turns 'Weak' (W) upon first step, then 'Very Weak' (V), and finally breaks into a 'Hole' (H). Stepping into a 'Hole' results in a high penalty and episode termination.
*   **Actions:** The agent has 8 discrete actions: Move (Left, Down, Right, Up) and Jump (Left, Down, Right, Up). Jumps move the agent two squares and consume a limited resource.
*   **Observation Space:** The state is fully observable and represented by a combination of the agent's spatial index, the number of jumps remaining, and a boolean indicating whether the treasure has been collected.
*   **Reward System:** Rewards are structured to encourage goal achievement (high positive for treasure/goal) and discourage undesirable actions (negative for movement costs, high penalty for falling into holes or invalid actions).

## Curriculum Learning in ArcticDashEnv

The project employs a Curriculum Learning approach, primarily focusing on how the agent adapts its strategy when faced with varying resource constraints (specifically, the `max_jumps` parameter).

*   **Progressive Training:** The agent is trained in a series of stages, starting with a higher `max_jumps` value (easier) and gradually decreasing it (making the problem more challenging).
*   **Knowledge Transfer:** Q-tables learned in an "easier" stage (more jumps) can be used to initialize the Q-table for a "harder" stage (fewer jumps). This allows the agent to adapt existing knowledge rather than learning from scratch, potentially accelerating convergence and fostering more sophisticated strategies.
*   **Strategic Adaptation:** The curriculum helps observe how the agent's optimal policy changes. For instance, with many jumps, the agent might take riskier, shorter paths, while with fewer jumps, it becomes more conservative, saving jumps for critical obstacles or the final return path.

## Authors

*   Kacper Machnik
*   Krzempek Michał

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
