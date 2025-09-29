import streamlit as st
import os
from kerykeion import AstrologicalSubject, NatalAspects, Report
import re
import json
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import subprocess
from dotenv import load_dotenv

load_dotenv()

# --- Astrological Data Functions (from astro.py) ---
def strip_think_tags(text: str) -> str:
    """
    Removes <think>...</think> tags and their content from a string.

    Args:
        text: The input string that may contain think tags.

    Returns:
        A new string with all think tags and their inner content removed.
    """
    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL)

def trim_astrological_report(full_report_text: str) -> str:
    """Trims the full astrological report to only major aspects for LLM analysis."""
    try:
        parts = full_report_text.split("## Natal Aspects")
        if len(parts) != 2:
            return full_report_text
        header = parts[0].strip()
        json_string = parts[1]
        aspects_list = json.loads(json_string)
    except (json.JSONDecodeError, IndexError):
        return full_report_text

    MAJOR_BODIES = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
    MAJOR_ASPECTS = ['conjunction', 'opposition', 'trine', 'square', 'sextile']
    ORB_THRESHOLD = 3.0
    
    important_aspect_summaries = []
    for aspect in aspects_list:
        p1, p2, aspect_type, orbit = aspect.get('p1_name'), aspect.get('p2_name'), aspect.get('aspect'), aspect.get('orbit')
        if (p1 in MAJOR_BODIES and p2 in MAJOR_BODIES and aspect_type in MAJOR_ASPECTS and abs(orbit) <= ORB_THRESHOLD):
            summary_line = f"- {p1} {aspect_type} {p2} (orb: {orbit:.2f}°)"
            important_aspect_summaries.append(summary_line)
            
    if not important_aspect_summaries:
        summary_section = f"## Key Natal Aspects\nNo major aspects found within a {ORB_THRESHOLD}° orb."
    else:
        summary_section = "## Key Natal Aspects\n" + "\n".join(important_aspect_summaries)
    return header + "\n\n" + summary_section


def generate_birth_chart_markdown(name, target_date, hour, minute, city, nation):
    """Generates a full birth chart report for a given date, time, and location."""
    try:
        subject = AstrologicalSubject(
            name=name, 
            year=target_date.year, 
            month=target_date.month, 
            day=target_date.day, 
            hour=hour, 
            minute=minute, 
            city=city, 
            nation=nation
        )
        report = Report(subject).get_full_report()
        aspects = NatalAspects(subject)
        aspects_data = [a.model_dump() for a in aspects.relevant_aspects]
        return f"{report}\n## Natal Aspects\n{json.dumps(aspects_data, indent=2)}"
    except Exception as e:
        return f"Could not generate birth chart for {name} in {city}, {nation}. Error: {e}"

# --- Helper Function to Get Ollama Models ---

def get_ollama_models():
    """Gets a list of available Ollama models."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            return []
        models = [line.split()[0] for line in lines[1:]]
        return models
    except FileNotFoundError:
        return [] # Ollama not installed
    except Exception:
        return [] # Other errors

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Caster")
st.title("Caster")

st.markdown("""
Welcome to the Astrological Sandbox! Add 'Agents' by providing their birth details. Once you have all your agents, generate their unique RPG classes. Finally, start the simulation to see how your characters interact in a narrated fantasy scenario.
""")

# --- Initialize Session State ---
if 'agents' not in st.session_state:
    st.session_state.agents = [
        {
            "name": "Agent 1",
            "date": datetime(2000, 1, 1),
            "hour": 12,
            "minute": 0,
            "city": "New York",
            "nation": "US"
        }
    ]
if 'rpg_classes' not in st.session_state:
    st.session_state.rpg_classes = {}
if 'story' not in st.session_state:
    st.session_state.story = ""


# --- Model and API Configuration ---
st.sidebar.header("1. Configure Your AI Model")

llm_provider = st.sidebar.selectbox("Select AI Provider:", ["Ollama", "Gemini"])

selected_model = None
gemini_api_key = os.getenv("GEMINI_API_KEY")

if llm_provider == "Ollama":
    ollama_models = get_ollama_models()
    if ollama_models:
        selected_model = st.sidebar.selectbox("Select an Ollama Model:", ollama_models)
    else:
        st.sidebar.warning("Ollama is not running or no models are installed. Please start Ollama and install a model (e.g., `ollama run llama3`).")
        st.stop()
elif llm_provider == "Gemini":
    if not gemini_api_key:
        st.sidebar.warning("GEMINI_API_KEY not found in .env file.")
    gemini_models = ["gemini-2.5-flash", "gemini-2.5-pro"] 
    selected_model = st.sidebar.selectbox("Select a Gemini Model:", gemini_models)


# --- Agent Management ---
st.header("2. Create Your Agents")

for i, agent in enumerate(st.session_state.agents):
    st.markdown(f"---")
    cols = st.columns([0.8, 0.2])
    cols[0].subheader(f"Agent {i+1}")
    if cols[1].button(f"Remove Agent {i+1}", key=f"remove_{i}"):
        st.session_state.agents.pop(i)
        if f"agent_{i+1}_class" in st.session_state.rpg_classes:
            del st.session_state.rpg_classes[f"agent_{i+1}_class"]
        st.rerun()

    agent['name'] = st.text_input("Agent Name:", value=agent.get('name', f'Agent {i+1}'), key=f"name_{i}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        agent['date'] = st.date_input("Birth Date:", value=agent['date'], key=f"date_{i}")
    with col2:
        agent['hour'] = st.number_input("Birth Hour (24h):", min_value=0, max_value=23, value=agent['hour'], step=1, key=f"hour_{i}")
    with col3:
        agent['minute'] = st.number_input("Birth Minute:", min_value=0, max_value=59, value=agent['minute'], step=1, key=f"minute_{i}")

    col4, col5 = st.columns(2)
    with col4:
        agent['city'] = st.text_input("Birth City:", value=agent['city'], key=f"city_{i}")
    with col5:
        agent['nation'] = st.text_input("Country (2-letter code):", value=agent['nation'], max_chars=2, key=f"nation_{i}")

if st.button("Add Agent"):
    new_agent_num = len(st.session_state.agents) + 1
    st.session_state.agents.append({
        "name": f"Agent {new_agent_num}",
        "date": datetime.now(),
        "hour": 12, "minute": 0,
        "city": "London", "nation": "GB"
    })
    st.rerun()

# --- Character Generation ---
st.header("3. Generate RPG Classes")

if st.button("Generate All RPG Classes", type="primary"):
    if not selected_model:
        st.error("Please select a valid model in the sidebar.")
    elif llm_provider == "Gemini" and not gemini_api_key:
        st.error("Please provide your Google API Key to use Gemini.")
    else:
        st.session_state.rpg_classes = {}
        st.session_state.story = "" # Reset story when re-generating classes
        
        try:
            llm = None
            if llm_provider == "Ollama":
                llm = Ollama(model=selected_model)
            elif llm_provider == "Gemini":
                llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=gemini_api_key)

            prompt_template = PromptTemplate(
                template="""
You are an Astrological Game Master. Your task is to create a complex and unique RPG class based on the provided astrological birth chart data for {agent_name}.

**Birth Chart Analysis for {agent_name}:**
{birth_chart}

**Your Task:**
Based on the astrological data above, create a compelling fantasy RPG class. The class should have the following components:

1.  **Class Name:** A creative and fitting name for the class. Base the class off five fundamental archtypes and give it a flavour and speciality based on the chart: warrior, priest, archer, wizard, sorcerer, rogue.
2.  **Summary:** An evocative description of the class, its role, and personality based on the patterns and signs of the chart. If present, analyze patterns like t-squares, yods, grand trines, etc.
3.  **Skills:** Detail how each significant astrological aspect or planetary position translates into a specific skill or ability. If a planet has no significant aspect, describe its standalone influence as a skill.
4.  **Philosophy and Weaknesses:** What makes this class unique? What are its core motivations, strengths, and inherent weaknesses?
5.  **Role within a Party:** How does this class function in a team? What are its synergies and potential conflicts with other archetypes?
6. **Image prompt**: An image prompt that visually depicts the class.
                """,
                input_variables=["agent_name", "birth_chart"]
            )
            
            chain = prompt_template | llm | StrOutputParser()

            for i, agent in enumerate(st.session_state.agents):
                agent_name = agent['name']
                with st.status(f"Generating class for {agent_name}...", expanded=True) as status:
                    full_chart = generate_birth_chart_markdown(agent_name, agent['date'], agent['hour'], agent['minute'], agent['city'], agent['nation'])
                    if "Could not generate" in full_chart:
                        st.error(full_chart)
                        status.update(label=f"Failed to generate chart for {agent_name}!", state="error")
                        continue
                    
                    trimmed_chart = trim_astrological_report(full_chart)
                    
                    rpg_class_output = chain.invoke({"agent_name": agent_name, "birth_chart": trimmed_chart})
                    rpg_class_output = strip_think_tags(rpg_class_output)
                    st.session_state.rpg_classes[f"agent_{i+1}_class"] = rpg_class_output
                    status.update(label=f"Class for {agent_name} generated!", state="complete")
            
            st.success("All RPG classes have been generated!")

        except Exception as e:
            st.error(f"An error occurred during generation: {e}")

# --- Display Generated Classes and Simulation Section ---
if st.session_state.rpg_classes:
    st.markdown("---")
    st.header("Generated Classes")
    for i, agent in enumerate(st.session_state.agents):
        class_key = f"agent_{i+1}_class"
        if class_key in st.session_state.rpg_classes:
            with st.expander(f"Show Class for {agent['name']}"):
                st.markdown(st.session_state.rpg_classes[class_key])

    # --- Simulation Section ---
    st.markdown("---")
    st.header("4. Simulation Sandbox")

    if st.button("Simulate Scenario", type="secondary"):
        with st.spinner("The narrator is weaving the next chapter..."):
            try:
                llm = None
                if llm_provider == "Ollama":
                    llm = Ollama(model=selected_model)
                elif llm_provider == "Gemini":
                    llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=gemini_api_key)

                narrator_prompt = PromptTemplate(
                    template="""
You are a Fantasy Story Weaver and Game Master. Your task is to write the next chapter of a story featuring a group of characters interacting to eeach other.,

**The Characters:**
{character_classes}

**The Story So Far:**
{story_history}

**Your Task:**
Write the next chapter of the story. 
- If the story is just beginning, introduce the characters and place them into a new, random fantasy scenario (e.g., a bustling market, a forgotten ruin, a tense negotiation).
- If continuing the story, build upon the previous events. Introduce a new challenge, a surprising twist, a good omen, or a meaningful character interaction.
- Ensure the characters act according to their astrological profiles and RPG class descriptions.
- Keep the chapter concise (2-4 paragraphs) and end on a note that invites continuation.
                    """,
                    input_variables=["character_classes", "story_history"]
                )
                
                narrator_chain = narrator_prompt | llm | StrOutputParser()

                # Format the character classes for the prompt
                all_class_descriptions = "\n\n---\n\n".join(st.session_state.rpg_classes.values())

                # Invoke the chain
                new_chapter = narrator_chain.invoke({
                    "character_classes": all_class_descriptions,
                    "story_history": st.session_state.story if st.session_state.story else "The story has not yet begun."
                })
                new_chapter = strip_think_tags(new_chapter)
                
                # Append the new chapter to the story
                st.session_state.story += f"\n\n## Chapter {len(st.session_state.story.split('## Chapter'))}\n\n" + new_chapter
            
            except Exception as e:
                st.error(f"An error occurred during simulation: {e}")

    if st.session_state.story:
        st.subheader("The Story So Far...")
        st.markdown(st.session_state.story)