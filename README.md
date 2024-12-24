# **SuperHero Call Center: Deep Q-Network (DQN) Model**

## **Overview**
This repository contains the implementation of a Deep Q-Network (DQN) model for the game **SuperHero Call Center**. The primary purpose of this model is to dynamically generate challenging and engaging crime events for players based on their performance. This ensures that gameplay remains balanced, stimulating, and tailored to the player’s skill level.

The DQN model uses reinforcement learning to evaluate the player's proficiency and create crime events that adapt to the evolving state of the game. It is trained using gameplay data, ensuring that the generated events provide a mix of variety and difficulty to maintain player engagement.

---

## **Project Structure**

```
SuperHeroCallCenter/
│
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   └── game_env.py          # Custom OpenAI Gym environment
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── dqn_agent.py         # DQN agent implementation
│   │   ├── network.py           # Neural network architecture
│   │   └── memory.py            # Replay memory implementation
│   │
│   ├── main.py                  # Main script to train and test the model
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py            # Hyperparameter configuration
│
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation

```

---

## **Installation**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/pawannitt-26/GameBot.git
   cd GameBot
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## **Configuration**
The file `config.py` contains all the hyperparameters and configurations for the model. Update the file to customize the training and gameplay setup.

### **Key Parameters:**
- **Replay Buffer:**
  - `buffer_size`: Maximum size of the replay buffer.
  - `batch_size`: Number of samples per training step.
- **Model Training:**
  - `learning_rate`: Learning rate for the optimizer.
  - `gamma`: Discount factor for future rewards.
  - `epsilon_start`, `epsilon_end`, `epsilon_decay`: Parameters for the epsilon-greedy exploration strategy.
- **Crime Event Generation:**
  - `max_tags_per_event`: Maximum tags in a single crime event.
  - `categories`: Available crime event categories and tags.

---

## **How It Works**

### **Model Workflow:**
1. **State Representation:**
   - The game environment sends states to the DQN model, including:
     - Player performance metrics.
     - Current crime events and their attributes.
     - Superheroes and their attributes.
     - City dynamics and crime levels.

2. **Action Selection:**
   - The DQN model suggests optimal difficulty and composition for the next crime event based on the player’s performance.
   - Actions correspond to generating new crime events with specific attributes (e.g., number of categories, tags, and difficulty).

3. **Reward Mechanism:**
   - Rewards are assigned based on the player’s interaction with the event (e.g., completion success, time taken, resources used).
   - Negative rewards for failed attempts or high resource expenditure.

4. **Crime Event Generation:**
   - Uses the selected action to generate a new event, ensuring it incorporates categories and tags that align with the player’s current level.

---

## **Examples**

### **Sample Input State:**
```json
{
  "player_level": 12,
  "crime_level": 65,
  "available_superheroes": ["HeroA", "HeroC"],
  "active_events": [
    {"id": 1, "difficulty": 8, "tags": ["Armed", "Hostage"], "timer": 120},
    {"id": 2, "difficulty": 5, "tags": ["Cyber Attack"], "timer": 300}
  ],
  "resources": {"money": 1000, "gadgets": 5},
  "public_morale": 80
}
```

### **Generated Crime Event:**
```json
{
  "id": 3,
  "difficulty": 9,
  "tags": ["Heist", "High-Tech", "Armed"],
  "zone": "Downtown",
  "impact": "Severe"
}
```

---
