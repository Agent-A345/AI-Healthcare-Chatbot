import streamlit as st
from transformers import pipeline
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="🏥",
    layout="wide"  
)

#CSS Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f8ff;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
        }
        .st-emotion-cache-16idsys p {
            font-size: 20px;
        }
        /* Fixed input container styling */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1rem;
            background-color: white;
            border-top: 1px solid #eee;
            z-index: 1000;  /* Ensure it stays on top */
        }
        /* Adjust main content to prevent overlap */
        .main .block-container {
            padding-bottom: 120px !important;  /* Increased padding */
            max-width: 100% !important;
        }
        /* Hide Streamlit's default footer */
        footer {
            visibility: hidden;
        }
        /* Ensure chat messages don't get cut off */
        .stChatMessageContent {
            overflow-wrap: break-word;
            word-wrap: break-word;
            hyphens: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Medical knowledge base for common questions
MEDICAL_KNOWLEDGE = {
    "flu": """Common flu symptoms include:
- Fever or feeling feverish/chills
- Cough and sore throat
- Runny or stuffy nose
- Muscle or body aches
- Headaches
- Fatigue (tiredness)
- Some people may have vomiting and diarrhea""",
    
    "fever": """Common fever symptoms include:
- Elevated body temperature (above 98.6°F/37°C)
- Chills and shivering
- Sweating
- Headache
- Muscle aches
- Loss of appetite
- Dehydration
- Weakness and fatigue""",
}

@st.cache_resource
def load_model():
    try:
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            model_kwargs={"device_map": "auto"}
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Fallback to CPU if GPU fails
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            device="cpu"
        )

def generate_medical_response(question, model):
    """Generate a medical response with fallback to knowledge base"""
    # Clean the question
    question = question.lower().strip()
    
    # Check knowledge base first
    for key, response in MEDICAL_KNOWLEDGE.items():
        if key in question:
            return response
    
    # If not in knowledge base, use the model
    prompt = f"""As a medical professional, provide a clear, accurate, and detailed response to the following question.
    Follow these guidelines:
    1. Start with a concise definition or overview if relevant
    2. List specific symptoms, signs, or key points in a structured manner
    3. Include important warning signs or red flags if applicable
    4. Mention when immediate medical attention is necessary
    5. Add preventive measures or self-care tips when appropriate
    
    Question: {question}
    
    Structure the response with appropriate headings and clear sections.
    Focus on medically verified information and avoid speculative advice.
    If the condition is serious, emphasize the importance of seeking professional medical care.
    
    Detailed medical response:"""
    
    try:
        response = model(
            prompt,
            max_length=500,
            min_length=100,
            temperature=0.7,
            repetition_penalty=1.9,
            do_sample=True
        )[0]['generated_text']
        
        # Clean up the response
        response = response.replace("Answer:", "").replace("Medical answer:", "").strip()
        return response
    except Exception as e:
        logger.error(f"Model error: {str(e)}")
        return "I apologize, but I'm having trouble generating a response. Please try rephrasing your question."

def format_response(response_text):
    return response_text

def main():
    st.markdown("<h1 style='text-align: center; color: #1e3d59; font-size: 42px;'>🏥 Healthcare Assistant Chatbot</h1>", unsafe_allow_html=True)
    
    # Load the model
    model = load_model()

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Create a container for chat messages
    chat_container = st.container()
    
    # Display chat messages in the container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Fixed input container at the bottom
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([5, 1])
        
        with col1:
            prompt = st.chat_input("Type your question here")
        
        with col2:
            if st.button("Clear Chat", type="primary"):
                st.session_state.messages = []
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.spinner("Generating response..."):
            response = generate_medical_response(prompt, model)
            formatted_response = format_response(response)

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
        with chat_container.chat_message("assistant"):
            st.write(formatted_response)

if __name__ == "__main__":
    main()
