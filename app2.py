import streamlit as st
from PyPDF2 import PdfReader
from faster_whisper import WhisperModel
from groq import Groq
import os
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from streamlit_mic_recorder import mic_recorder
import io
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


load_dotenv()


def translator(client, text_input, detected_language, model_name):
  prompt = f'''
    You are a language translator, you'll get a text input : {text_input} and the detected language of the text :  {detected_language}. Your job is to translate the text input into english. For example :

    When the detected language is French, and the text input is 'J'aime programmer' then the output in English is supposed to be 'I love Programming'.

    When you're giving the output, only include input text and translation.

  '''.format(text_input = text_input, detected_language=detected_language)
  return groq_chat(client, prompt, model_name, None)

def groq_chat(client, prompt,  model, response_format):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        response_format=response_format
    )
    return completion.choices[0].message.content



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text




def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="\n",
        chunk_size=150,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore




def get_relevant_excerpts(user_question, vectorstore):

  relevent_docs = vectorstore.similarity_search(user_question)
  return relevent_docs






def main():

  st.set_page_config(page_title="Chat with Audio",
                       page_icon=":mic:")
  st.write(css, unsafe_allow_html=True)
  






  os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
  model_size = 'medium'





  if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
  if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
  if 'text_received' not in st.session_state:
    st.text_received = []
  if not '_last_speech_to_text_transcript' in st.session_state:
        st.session_state._last_speech_to_text_transcript = None
  


  
  st.header("Your Tensergo Voice Chat")
  audio = mic_recorder(start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=False, key='recorder', format='webm')
  st.write(user_template.replace("{{MSG}}", 'Hello Bot'), unsafe_allow_html=True)
  st.write(bot_template.replace("{{MSG}}", 'Hello Human, First add the file and Click on Process, thereafter HAPPY RECORDING!'), unsafe_allow_html=True)


  if audio and st.session_state.vectorstore:
      # id = audio['id']
      # new_output = (id > st.session_state._last_speech_to_text_transcript_id)
      # if new_output:
      #     output = None
      #     st.session_state._last_speech_to_text_transcript_id = id
      audio_bio = io.BytesIO(audio['bytes'])
      audio_bio.name = 'audio.webm'
      st.audio(audio['bytes'], format='audio/webm')

      with open('recorded_audio','wb') as f:
          f.write(audio['bytes'])

      

      model = WhisperModel(model_size, device="cpu", compute_type="int8", local_files_only=True)

      segments, info = model.transcribe(audio_bio, beam_size=5)

      print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

      final_text = ''
      for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        final_text += segment.text

    #   st.write(final_text)


      

      model_name = 'mixtral-8x7b-32768'
      groq_api_key = os.getenv('GROQ_API_KEY')


      # text_input = final_text
      detected_language = info.language

      client = Groq(
              api_key = groq_api_key
          )

      text_input = translator(client, final_text, detected_language, model_name)
    #   st.write(text_input)
      st.session_state.chat_history.append({"role": "user", "content": f'User Audio Text : {final_text}{text_input}'})



      relevant_excerpts = get_relevant_excerpts(text_input, st.session_state.vectorstore)
      excerpt_final = ""
      for excerpt in relevant_excerpts:
            excerpt_final += excerpt.page_content


      with open('base_prompts_tensorgo.txt', 'r') as file:
            base_prompt = file.read()

      full_prompt =  base_prompt.format(text_input=text_input, provided_excerpts=excerpt_final)

      llm_response =  groq_chat(client, full_prompt, model_name, None )
    #   st.write(llm_response)
      st.session_state.chat_history.append({"role": "assistant", "content": llm_response})

      for i, message in enumerate(st.session_state.chat_history):
            # st.write(message)
            # st.write(i,message)
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message['content']), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message['content']), unsafe_allow_html=True)












  with st.sidebar:
      st.subheader("Upload File")
      pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
      if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create vector store
                st.session_state.vectorstore = get_vectorstore(text_chunks)


  

if __name__ == "__main__":
    main()
