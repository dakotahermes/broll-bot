from pydantic import BaseModel
from typing import List, Literal
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Step 1: Define input structure
class AdScriptInput(BaseModel):
    script: str
    tone: Literal['inspiring', 'urgent', 'calm', 'funny']
    format: Literal['UGC', 'talking_head', 'testimonial']

# Step 2: Define output structure for b-roll suggestions
class SceneBeat(BaseModel):
    timestamp: str
    scene_description: str
    emotion: str
    script_excerpt: str

class BrollPrompt(BaseModel):
    prompt: str
    duration: int
    aspect_ratio: str
    insert_after: str

# Agent 1: Uses OpenAI to parse the script into emotional scene beats
# Avoid mentioning speakers or any direct camera shots
def agent_1_parse_script(ad_input: AdScriptInput) -> List[SceneBeat]:
    system_prompt = """
    You are a video editor assistant. Given a script, break it into a list of emotional scene beats for background B-roll visuals.
    Do not mention the speaker, camera shots, or the narrator directly.
    Each beat should:
    - Suggest only ambient or illustrative scenes
    - Include a timestamp (like 00:00, 00:05, etc.)
    - Describe the visual scene
    - Include the emotion being conveyed
    - Quote the part of the script this visual relates to ("script_excerpt")

    Output JSON in this format:
    [
        {"timestamp": "00:00", "scene_description": "A dark, empty room with rain outside the window", "emotion": "hopeless", "script_excerpt": "I felt completely alone..."},
        ...
    ]
    """

    user_prompt = f"""
    Script: {ad_input.script}
    Tone: {ad_input.tone}
    Format: {ad_input.format}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    beats_data = json.loads(response.choices[0].message.content)
    return [SceneBeat(**beat) for beat in beats_data]

# Agent 2: Converts scene beats into video generation prompts
def agent_2_generate_prompts(beats: List[SceneBeat], duration: int = 5, aspect_ratio: str = "9:16") -> List[BrollPrompt]:
    prompts = []
    for beat in beats:
        formatted = f"{beat.scene_description}, cinematic, {beat.emotion} mood"
        prompts.append(BrollPrompt(prompt=formatted, duration=duration, aspect_ratio=aspect_ratio, insert_after=beat.script_excerpt))
    return prompts

# Streamlit UI
st.set_page_config(page_title="B-Roll Bot", layout="centered")
st.title("🎬 B-Roll Bot")

script_input = st.text_area("Paste your ad script here:")
tone = st.selectbox("Select Tone:", ["inspiring", "urgent", "calm", "funny"])
format_type = st.selectbox("Select Format:", ["UGC", "talking_head", "testimonial"])

if st.button("Generate B-Roll Prompts") and script_input:
    ad_input = AdScriptInput(script=script_input, tone=tone, format=format_type)
    scene_beats = agent_1_parse_script(ad_input)
    broll_prompts = agent_2_generate_prompts(scene_beats)

    st.subheader("Generated Prompts")
    for prompt in broll_prompts:
        st.markdown(f"**Insert after:** _{prompt.insert_after}_")
        st.json({"prompt": prompt.prompt, "duration": prompt.duration, "aspect_ratio": prompt.aspect_ratio})
