# AI Healthcare Chatbot
This project is an AI-powered healthcare chatbot designed to assist users by answering health-related queries, analyzing symptoms, and providing preliminary medical insights. It integrates a pre-trained AI model (Flan-T5-Large) along with a medical knowledge base to enhance response accuracy.

## Features
+ Provides AI-generated responses to health-related queries
+ Uses a predefined medical knowledge base for instant answers
+ Utilizes natural language processing (NLP) for better understanding
+ Built using the Streamlit framework for a user-friendly interface
+ Includes logging and error handling for debugging and monitoring

## Installation and Setup
1. **Clone the Repository**<br>
   git clone https://github.com/Agent-A345/AI-Healthcare-Chatbot.git<br> 
   cd AI-Healthcare-Chatbot<br>   
2. **Create a Virtual Environment (Optional)**<br>
   python -m venv chatbot_env<br>
   
    **Activate the virtual environment:**<br>
    + **On macOS/Linux:**   source chatbot_env/bin/activate<br>
   + **On Windows:**        chatbot_env\Scripts\activate<br>
3. **Install Dependencies**<br>
   pip install -r requirements.txt<br>
4. **Run the Chatbot**<br>
   streamlit run app.py<br>

## Technologies Used
+ Python 3.x
+ Streamlit (For user interface)
+ Transformers (For AI model integration)
+ PyTorch (For deep learning)
+ Logging module (For error handling and debugging)

## How It Works
+ The user enters a health-related query in the chatbot.
+ The chatbot first checks the predefined medical knowledge base for an instant response.
+ If the query is not found in the knowledge base, it is processed by the Flan-T5-Large model.
+ The model generates a context-aware medical response using NLP techniques.
+ The response is displayed in the Streamlit UI.

## Future Enhancements
+ Implementing voice input for hands-free interaction
+ Adding multilingual support for better accessibility
+ Integrating with external medical APIs for real-time data retrieval
+ Developing a mobile application for better accessibility

## License
This project is licensed under the Apache License 2.0
