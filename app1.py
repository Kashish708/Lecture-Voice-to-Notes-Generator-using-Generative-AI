import streamlit as st
import os
import json
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from typing import List


from dotenv import load_dotenv
load_dotenv() 


class Flashcard(BaseModel):
    """Defines the structure for a single flashcard item."""
    concept: str = Field(description="The key concept or term from the lecture.")
    definition: str = Field(description="The detailed definition or explanation (the back of the card).")

class LectureFlashcards(BaseModel):
    """Defines the structure for the full set of flashcards."""
    title: str = Field(description="A title for the flashcard set.")
    flashcards: List[Flashcard] = Field(description="A list of 10 generated flashcards.")

class QuizQuestion(BaseModel):
    """Defines the structure for a single quiz question."""
    question: str = Field(description="The question based on the lecture content.")
    options: List[str] = Field(description="A list of 4 possible answers.")
    correct_answer: str = Field(description="The correct answer text from the options list.")

class LectureQuiz(BaseModel):
    """Defines the structure for the full set of quiz questions."""
    title: str = Field(description="A title for the generated quiz.")
    questions: List[QuizQuestion] = Field(description="A list of 5 generated quiz questions.")




@st.cache_resource
def get_ai_client():
    """Initializes and returns the Gemini client using environment variable."""
  
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    
    if not api_key:
        
        st.error("🚨 AI API Key Missing!")
        st.warning(
            "The app could not find the API key. Please ensure you have a **.env** file "
            "in the same directory as this script, containing your key like this: \n"
            "```\nGEMINI_API_KEY=\"Your_Actual_Key_Here\"\n```"
        )
        return None
        
    try:
       
        client = genai.Client()

        client.models.list() 
        return client
    except Exception as e:
        st.error(f"❌ Initialization Failed!")
        st.warning(f"Error connecting to the API. Check your key is valid.")
        st.code(f"Underlying error: {str(e)}")
        return None

def transcribe_audio_and_time_sync(client: genai.Client, audio_file_path: str):
    """Transcribes the audio and forces the model to include time markers."""
    st.info("Step 1/4: Uploading and transcribing audio with time synchronization...")
    
    audio_file = client.files.upload(file=audio_file_path)
    
    system_prompt = (
        "You are an expert transcriber. Transcribe the lecture verbatim. "
        "Insert a precise timestamp `[HH:MM:SS]` at the start of every new sentence "
        "or after any significant pause (at least 2 seconds). Output must be pure time-synced text."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[audio_file, "Please transcribe the entire lecture audio and format the output as requested."],
        config=types.GenerateContentConfig(system_instruction=system_prompt)
    )

    client.files.delete(name=audio_file.name)
    
    return response.text

def generate_study_content(client: genai.Client, transcript: str, content_type: str, step_number: int):
    """Generates structured study content based on the transcript."""
    
    st.info(f"Step {step_number}/4: Generating {content_type}...")

    if content_type == "Summary Notes":
        
        system_prompt = (
            "You are an academic note-taker. Convert the time-synced transcript into "
            "clear, well-structured study notes using Markdown (headings, bolding, bullet points). "
            "DO NOT include the timestamps in the final notes. Focus on clarity and key concepts."
        )
        prompt = "Generate comprehensive study notes based on this lecture transcript:\n\n" + transcript
        config = types.GenerateContentConfig(system_instruction=system_prompt)
        
        response = client.models.generate_content(model='gemini-2.5-flash', contents=[prompt], config=config)
        return "markdown", response.text
    
    schema_map = {
        "Quiz": {"prompt": "Generate a challenging 5-question multiple-choice quiz.", "schema": LectureQuiz},
        "Flashcards": {"prompt": "Generate 10 key flashcards (concept and definition).", "schema": LectureFlashcards}
    }
    
    selected = schema_map.get(content_type)
    
    system_prompt = f"You are an expert content creator. Generate content strictly following the JSON schema provided. The content must be derived directly from the lecture transcript."

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=selected["schema"],
    )
    
    prompt = selected["prompt"] + "\n\nTranscript:\n" + transcript
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt],
        config=config
    )
    
    return "json", json.loads(response.text)




def display_quiz_results(quiz_data):
    """Displays the generated quiz data interactively."""
    st.subheader(f"🧠 {quiz_data.get('title', 'Generated Quiz')}")
    st.markdown("Try to answer the questions below!")
    
    for i, q in enumerate(quiz_data.get('questions', [])):
        options = q.get('options', [])
        
        with st.expander(f"Question {i+1}: {q.get('question', 'Missing Question')}", expanded=False):
            user_choice = st.radio(
                "Select your answer:",
                options,
                key=f"quiz_{i}",
                index=None
            )
            
            if user_choice:
                correct_answer = q.get('correct_answer', 'N/A')
                if user_choice == correct_answer:
                    st.success("✅ Correct! You nailed it.")
                else:
                    st.error(f"❌ Incorrect. The correct answer was: **{correct_answer}**")

def display_flashcard_results(card_data):
    """Displays the generated flashcards using expanders as flip-cards."""
    st.subheader(f"📝 {card_data.get('title', 'Generated Flashcards')}")
    st.info("Click on a card's concept to reveal the definition.")
    
    flashcards = card_data.get('flashcards', [])
    cols = st.columns(3) # Display 3 cards per row

    for i, card in enumerate(flashcards):
        with cols[i % 3]:
            
            with st.container(border=True):
                
                if st.session_state.get(f'card_{i}_flipped', False):
                    st.markdown(f"**Concept:** {card['concept']}")
                    st.markdown(f"**Definition:** {card['definition']}")
                    if st.button("Flip Back", key=f"card_btn_back_{i}", use_container_width=True):
                        st.session_state[f'card_{i}_flipped'] = False
                        st.rerun()
                else:
                    st.markdown(f"**Concept {i+1}:**", help="Click to reveal definition")
                    st.markdown(f"***{card['concept']}***")
                    if st.button("Flip to Definition", key=f"card_btn_flip_{i}", use_container_width=True):
                        st.session_state[f'card_{i}_flipped'] = True
                        st.rerun()




def main():
    st.set_page_config(page_title="Lecture AI Assistant", layout="wide")
    st.title("🎤 Lecture Voice-to-Notes Generator")
    st.markdown("Upload audio to generate a **time-synced transcript**, structured **Summary Notes**, an interactive **Quiz**, and helpful **Flashcards**.")

    
    with st.sidebar:
        st.header("Lecture Input")
        uploaded_file = st.file_uploader(
            "Upload Audio File (MP3/WAV/M4A)", 
            type=['mp3', 'wav', 'm4a'],
            help="Audio files must be stored locally for the AI to process. Max recommended size: ~10 minutes."
        )

        st.markdown("---")
        process_button = st.button("Generate All Study Content", type="primary", use_container_width=True)

    
    if process_button:
        if not uploaded_file:
            st.error("Please upload an audio file to begin.")
            return

        client = get_ai_client()
        
        if not client: return

        
        temp_audio_dir = "temp_audio_file"
        temp_audio_path = os.path.join(temp_audio_dir, uploaded_file.name)
        os.makedirs(temp_audio_dir, exist_ok=True) 

        
        try:
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            
            with st.status("Processing lecture and generating content...", expanded=True) as status:
                
                
                transcript = transcribe_audio_and_time_sync(client, temp_audio_path)
                st.session_state['transcript'] = transcript

                
                _, notes_content = generate_study_content(client, transcript, "Summary Notes", 2)
                st.session_state['notes'] = notes_content

            
                _, quiz_content = generate_study_content(client, transcript, "Quiz", 3)
                st.session_state['quiz'] = quiz_content

                
                _, flashcard_content = generate_study_content(client, transcript, "Flashcards", 4)
                st.session_state['flashcards'] = flashcard_content
                
            status.update(label="All study materials generated successfully!", state="complete", expanded=False)
            st.toast("Processing complete! Your materials are ready.", icon='🚀')
            st.rerun() 

        except Exception as e:
            st.error(f"An unexpected error occurred during content generation: {e}")
            st.warning("Ensure the audio quality is good and the content is clearly spoken.")
        finally:
            
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    
    if 'transcript' in st.session_state:
        
        
        st.markdown("---")
        st.header("1. Original Time-Synced Transcript")
        st.info("The AI included timestamps to link notes to key moments in the audio.")
        st.code(st.session_state['transcript'], language='markdown')

        
        st.markdown("---")
        st.header("2. Summary Notes")
        st.markdown(st.session_state['notes'])

        
        st.markdown("---")
        st.header("3. Interactive Study Tools")
        
        quiz_col, card_col = st.columns(2)
        
        with quiz_col:
            display_quiz_results(st.session_state['quiz'])
            
        with card_col:
            display_flashcard_results(st.session_state['flashcards'])

        


if __name__ == "__main__":
    main()

