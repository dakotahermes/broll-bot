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
    tone: Literal['inspiring', 'urgent', 'calm', 'funny', 'serious', 'emotional', 'uplifting', 'mysterious']
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
    search_instruction: str

# Agent 1: Uses OpenAI to parse the script into emotional scene beats
# Avoid mentioning speakers or any direct camera shots
def agent_1_parse_script(ad_input: AdScriptInput) -> List[SceneBeat]:
    system_prompt = """
You are a video editing assistant.

Given a video script, break it into a list of B-roll moments designed to visually support the tone, format, and emotion of the message.

For each moment, output:
- A timestamp (e.g., 00:00, 00:05, etc.)
- A vivid, descriptive scene (ambient, symbolic, illustrative, or emotional)
- The core emotion the visual supports
- A short excerpt from the script that the scene should follow (for placement)

Guidelines:
- Use visuals that match or deepen the emotional tone (e.g., hopeful, mysterious)
- Avoid referencing direct speakers or narrators unless appropriate to the format (e.g., UGC)
- You may use symbolic or metaphorical imagery when relevant
- Be creative but stay relevant to the script's meaning and tone
- Do not refer to specific character names or identities (e.g., “Spencer” or “she”) — describe people generically (e.g., “a man,” “a woman,” “a musician”)


Respond only in raw JSON. Do not include markdown or explanation.

Example format:
[
  {
    "timestamp": "00:00",
    "scene_description": "Fog drifting through a forest at dawn",
    "emotion": "mysterious",
    "script_excerpt": "I didn’t know what I was searching for..."
  },
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

# Agent 2: Converts scene beats into video generation prompts + search instructions
def agent_2_generate_prompts(beats: List[SceneBeat], duration: int = 5, aspect_ratio: str = "9:16") -> List[BrollPrompt]:
    prompts = []
    for beat in beats:
        review = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're an AI video generation advisor. Given a scene description, judge whether a generative video AI like Kling or Pika could realistically generate this scene effectively. Be strict. Only approve if it's visually specific, feasible with current generative tools, and not too abstract or complex. Respond only with 'yes' or 'no'."},
                {"role": "user", "content": beat.scene_description}
            ]
        )
        verdict = review.choices[0].message.content.strip().lower()
        if verdict == "yes":
            formatted = f"{beat.scene_description}, cinematic, {beat.emotion} mood"
            search_instruction = f"Search stock or AI video libraries for: '{beat.scene_description}' with a {beat.emotion} vibe."
            prompts.append(BrollPrompt(
                prompt=formatted,
                duration=duration,
                aspect_ratio=aspect_ratio,
                insert_after=beat.script_excerpt,
                search_instruction=search_instruction
            ))
    return prompts

# Streamlit UI
st.set_page_config(page_title="B-Roll Bot", layout="centered")
st.title("🎬 B-Roll Bot")

script_input = st.text_area("Paste your ad script here:")
tone = st.selectbox("Select Tone:", ["inspiring", "urgent", "calm", "funny", "serious", "emotional", "uplifting", "mysterious"])
format_type = st.selectbox("Select Format:", ["UGC", "talking_head", "testimonial"])

if st.button("Generate B-Roll Prompts") and script_input:
    ad_input = AdScriptInput(script=script_input, tone=tone, format=format_type)
    scene_beats = agent_1_parse_script(ad_input)
    broll_prompts = agent_2_generate_prompts(scene_beats)

    st.subheader("Generated Prompts")
    for prompt in broll_prompts:
        st.markdown(f"**Insert after:** _{prompt.insert_after}_")
        st.markdown(f"**Search Tip:** {prompt.search_instruction}")
        st.json({"prompt": prompt.prompt, "duration": prompt.duration, "aspect_ratio": prompt.aspect_ratio})
