# Caster: The Ultimate Social Engineering Handbook for Social Rogues and Hackers

---

### The only way to undo a monster without becoming it: is by understanding it

### **A Word from the Creator**

> "I grew up playing RPGs. The idea of archetypes, classes, strengths, and weaknesses shaping a character's journey always fascinated me. 'Caster' is born from that nostalgia, fused with an unorthodox exploration of human nature. It's a tool for 'people profiling'—to anticipate what others might do before they do it. The profiling is eclectic, unorthodox, and requires a leap of faith. I invite you to suspend disbelief, embrace the theme, and see what you discover."

---

## What is Caster?
<img width="1024" height="1024" alt="image (2)" src="https://github.com/user-attachments/assets/158c21a8-3397-4604-8add-2111a2a038aa" />

Caster is not your typical personality profiler. It's a sandbox for understanding hypothetical scenarios through the lens of astrology and RPG archetypes. Think of it as a "social engineering" toolkit for the modern-day rogue, hacker, or anyone curious about the hidden patterns in human behavior. Each day or human situation is a monster waiting to be "mastered": people present themselves as challenges, not knowing that the question they want answered about themselves reflected in another person, is deeply idiosincratic _to you_: **can you master me? can you become "me" and ride this wave?** 

The best way to deal with situations, is by being resonant with them: don't oppose them, but let them teach you, so you can eventually ride the beast.

_Inspired in King David's adventures as a rogue, we encourage you yo become sovereign and to not project your own bullshit unto other people: each day is a test, and deciding what to do about the test is about you and you only. **Not about other people** — the safest way of being, is your memory of what you did. No one can ever strip you away from that privilege._


https://m.youtube.com/watch?v=nBHEX0odIMY&pp=ygUeSG93IHRvIGJlIGEgY29tc2VydmF0aXZlIHJlYmVs


If you are not interested in the people profiling aspect, you can use _caster_ as a means to know what to expect from a day personfied as an archetype: esentially each day has a blueprint. On the basis of astrological belief, if you do not become _one with the day_ you'll be punished for a lack of adaptation. This application is designed for those who believe that unconventional methods can yield unique insights. It's for the creatively-minded, the strategically inclined, and anyone who enjoys exploring the "what ifs" of human interaction.

## The Philosophy: A Leap of Faith into Unorthodox Profiling

Caster is built on an eclectic foundation that blends ancient astrological data with the imaginative framework of role-playing games. We acknowledge that this is an unorthodox approach. It requires a willingness to explore correlations and patterns that traditional data analysis might overlook.

We're not claiming to predict the future or read minds. Instead, Caster offers a unique framework for:

*   **Anticipatory Profiling:** Gaining a creative edge by considering a person's potential archetypal strengths and weaknesses.
*   **Social Hacking:** Understanding the "human algorithm" from a different perspective to improve communication, influence, and strategy.
*   **The Social Rogue's Handbook:** Providing a playful yet insightful tool for those who navigate the world with wit, charm, and a healthy dose of skepticism.

## Features

*   **Agent Creation:** Build profiles for individuals ("Agents") using their birth date, time, and location.
*   **Astrological Chart Generation:** Caster generates a detailed astrological chart for each agent, providing the raw data for their profile.
*   **AI-Powered RPG Class Generation:** Leveraging the power of LLMs (like Ollama and Gemini), Caster translates astrological data into unique, fantasy-themed RPG classes. Each class includes:
    *   A creative **Class Name** based on classic RPG archetypes.
    *   A **Summary** of the character's core traits and motivations.
    *   A list of **Skills** derived from specific astrological aspects.
    *   An exploration of the class's **Philosophy and Weaknesses**.
    *   An analysis of their **Role within a Party** (or team).
*   **Simulation Sandbox:** Place your generated characters into a dynamic, AI-narrated fantasy scenario. See how their archetypes interact, clash, and collaborate.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Pip
*   An environment that can run Streamlit applications.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/iblameandrew/caster.git
    cd caster
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables:**
    *   Create a `.env` file in the root directory.
    *   If you plan to use the Gemini models, add your Google API key to the `.env` file:
        ```
        GEMINI_API_KEY="your_google_api_key"
        ```

### Running the Application

1.  **Run the Streamlit app from your terminal:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your web browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## How to Use Caster: A Social Rogue's Guide

1.  **Configure Your AI Model:** In the sidebar, choose your preferred AI provider (Ollama or Gemini) and select a model. For Ollama, ensure it is running on your local machine.

2.  **Create Your Agents:**
    *   The application starts with one default agent.
    *   Enter the `name`, `birth date`, `time`, `city`, and `country` (2-letter code) for your "target" or team member.
    *   Add as many agents as you need for your analysis.

3.  **Generate RPG Classes:**
    *   Click the "Generate All RPG Classes" button.
    *   The AI will analyze each agent's astrological data and create a detailed RPG class profile. This is the core of the "people profiling" engine.

4.  **Simulate a Scenario:**
    *   Once the classes are generated, you can run a simulation.
    *   Click "Simulate Scenario" to have the AI weave a narrative featuring your agents. This is where you can observe their archetypal behaviors in action.

## A Note on the "Eclectic" Nature of Caster

The connection between astrology and personality is a subject of belief, not empirical science. Caster is presented as a tool for creative exploration and imaginative profiling. It's about looking at people and situations through a different lens, sparking new ideas and perspectives that you might not have considered otherwise.

Whether you're a writer looking for character inspiration, a team leader trying to understand group dynamics from a fresh angle, or a "social rogue" sharpening your intuitive skills, Caster offers a unique and engaging sandbox.

## Contributing

We welcome contributions from fellow social rogues, developers, and creative thinkers. If you have ideas for new features, improvements, or bug fixes, please feel free to open an issue or submit a pull request.
