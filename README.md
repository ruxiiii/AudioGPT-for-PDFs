# AudioGPT-for-PDFs
AudioGPT for PDFs is an AI-powered application that transcribes audio inputs, translates them, and provides answers based on PDF content. It uses the Whisper model for transcription from multiple lanuages, Groq for language translation and response generation, and FAISS for semantic search within PDF documents.

## Features 
* Audio Transcription: Converts spoken language into text using the Whisper model.
* Language Translation: Translates transcribed text to English using Groq.
* PDF Content Retrieval: Searches for relevant excerpts within uploaded PDF documents.
* Intelligent Responses: Generates answers based on the provided excerpts.

## Getting Started 
1. Clone the repository :
   ```
    git clone https://github.com/ruxiiii/AudioGPT-for-PDFs.git
    cd audiogpt-for-pdfs
   ```

2. Install the required packages:
   ```
     pip install -r requirements.txt
   ```

3. Download the required faster-whisper model :

     Download the faster-whisper-medium model from the Systran repository and place it in the appropriate directory.

4. Add your API keys:

   Create a .env file in the project directory and add your Groq API key:
   ```
     GROQ_API_KEY=your_groq_api_key
   ```

5. Run the app :
   ```
     streamlit run app2.py
   ```

## Usage 
1. Upload your PDF files via the sidebar and click "Process".
2. Use the microphone button to record your audio. The app will transcribe, translate, and retrieve relevant excerpts from the PDFs to provide a response.


   
