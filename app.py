import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
# from langchain.memory import ChatMessageHistory
# from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# Disease Information - All data as single text blocks for vector storage
DISEASE_KNOWLEDGE = [
    """Disease: Bacterial Blight
Severity: High
Treatment: Apply copper bactericides immediately and remove all affected leaves from the plant. This is a serious bacterial infection that can spread rapidly.
Confidence Level: When detected with high confidence above 80 percent, immediate action required.
Urgent Actions: Apply treatment within 24 hours. Monitor daily for changes. Remove infected plant material immediately. Consider expert consultation if spread continues.
Prevention: Maintain proper plant spacing, avoid overhead watering, use disease-free seeds.""",

    """Disease: Curl Virus
Severity: Very High
Treatment: Remove infected plants completely and control whiteflies which spread this virus. No cure available once infected.
Confidence Level: When detected with confidence above 70 percent, immediate plant removal necessary.
Urgent Actions: Apply treatment within 24 hours. Monitor daily for new infections. Control whitefly population immediately. Consider expert consultation for field management.
Prevention: Control whitefly vectors, use virus-resistant varieties, maintain field hygiene.""",

    """Disease: Healthy Leaf
Severity: None
Treatment: Continue good agricultural practices. No treatment needed.
Confidence Level: When detected with confidence above 90 percent, indicates good plant health.
Maintenance: Continue current practices. Regular monitoring for early disease detection. Maintain proper nutrition and watering.""",

    """Disease: Herbicide Damage
Severity: Medium
Treatment: Stop herbicide use immediately and increase watering to help plant recovery. Provide supportive care.
Confidence Level: When detected with confidence above 75 percent, indicates chemical damage.
Recommended Actions: Apply treatment within 2-3 days. Monitor weekly for recovery. Adjust irrigation schedule. Avoid herbicide application near affected area.""",

    """Disease: Leaf Hopper
Severity: Medium
Treatment: Apply appropriate insecticide and remove weeds around plants which harbor leaf hoppers.
Confidence Level: When detected with confidence above 70 percent, indicates pest infestation.
Recommended Actions: Apply treatment within 2-3 days. Monitor weekly for pest population. Remove weed hosts. Use integrated pest management.""",

    """Disease: Leaf Redding
Severity: Low
Treatment: Apply potassium fertilizer to address nutrient deficiency causing leaf reddening.
Confidence Level: When detected with confidence above 80 percent, indicates potassium deficiency.
General Care: Apply treatment when convenient. Monitor bi-weekly for improvement. Soil test recommended for proper fertilization.""",

    """Disease: Leaf Variegation
Severity: Low
Treatment: Monitor symptoms and provide supportive care. Usually indicates minor stress or genetic variation.
Confidence Level: When detected with confidence above 75 percent, indicates minor plant stress.
General Care: Apply treatment when convenient. Monitor bi-weekly for changes. Maintain consistent watering and nutrition."""
]

# General cotton care knowledge
GENERAL_KNOWLEDGE = [
    """Cotton Disease Prevention and Management
Prevent cotton diseases through proper crop rotation every 3-4 years. Maintain adequate spacing between plants for air circulation. Conduct regular field inspection weekly during growing season. Remove infected plant material immediately. Maintain proper soil drainage and nutrition. Use certified disease-free seeds. Practice integrated pest management.""",

    """Early Disease Detection in Cotton
Early detection saves crops and reduces losses. Look for leaf spots, wilting, discoloration, unusual growth patterns, and pest damage. Check plants weekly during growing season. Pay attention to weather conditions that favor disease development. Use mobile apps and AI tools for quick identification. Document symptoms with photos for expert consultation.""",

    """Cotton Treatment Application Guidelines
Follow label instructions carefully for all treatments. Use appropriate protective equipment during application. Apply during calm weather conditions with proper temperature. Ensure complete coverage of affected areas. Time applications based on disease cycle and weather. Keep records of treatments applied. Monitor effectiveness and adjust if needed."""
]

CLASS_NAMES = ["Bacterial Blight", "Curl Virus", "Healthy Leaf", "Herbicide Damage", "Leaf Hopper", "Leaf Redding", "Leaf Variegation"]

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def create_vector_store():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        documents = []
        
        # Add disease knowledge
        for i, knowledge in enumerate(DISEASE_KNOWLEDGE):
            doc = Document(
                page_content=knowledge,
                metadata={"source": "disease_database", "id": i}
            )
            documents.append(doc)
        
        # Add general knowledge
        for i, knowledge in enumerate(GENERAL_KNOWLEDGE):
            doc = Document(
                page_content=knowledge,
                metadata={"source": "general_knowledge", "id": i}
            )
            documents.append(doc)
        
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def setup_rag_chain(groq_api_key, retriever):
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct", # mixtral-8x7b-32768 , llama-3.3-70b-versatile
            temperature=0.1
        )            
        
        contextualize_q_system_prompt = (
            "Given the chat history and the latest user question about cotton diseases, "
            "create a standalone question that can be understood without the chat history."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        # to include past chat history
        history_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        system_prompt = (
            "You are a cotton disease specialist. Use the retrieved context to provide clear, practical answers about cotton diseases, treatments, and care. "
            "Give specific actionable advice. If unsure, say so and suggest consulting agricultural experts. "
            "Keep responses simple and focused.\n\nContext:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_retriever, qa_chain)
        
        return RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    except Exception as e:
        st.error(f"Error setting up RAG chain: {str(e)}")
        return None

@st.cache_resource
def load_model():
    model_path = "cotton_leaf_cnn_best_model.keras"
    
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        return None
    
    try:
            import keras
            model = keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            st.warning("Model loaded using Keras directly")
            return model
    except Exception as keras_error:
            st.error(f"Keras loading failed: {keras_error}")
            return None

def preprocess_image(image):
    image = image.resize((224, 224))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_disease(model, image):
    if model is None:
        return None, None
    
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        disease = CLASS_NAMES[predicted_idx]
        return disease, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def create_detection_text(disease, confidence):
    disease_info = next((d for d in DISEASE_KNOWLEDGE if disease in d), None)
    result_text = f"Current Detection Result: Disease detected as {disease} with confidence level of {confidence:.1%}. "
    if disease_info:
        result_text += disease_info
    return result_text

# Main App
st.title("Cotton Disease Detector with AI Assistant")
st.write("Upload cotton leaf images for disease detection and ask questions")

# Sidebar
with st.sidebar:
    st.header("Setup")
    
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    if st.button("Setup Vector Database"):
        with st.spinner("Setting up database..."):
            st.session_state.vectorstore = create_vector_store()
            if st.session_state.vectorstore:
                st.success("Database ready")
    
    if st.session_state.model is None:
        if st.button("Load Disease Model"):
            with st.spinner("Loading model..."):
                st.session_state.model = load_model()
                if st.session_state.model:
                    st.success("Model loaded")
    
    if st.session_state.vectorstore and groq_api_key and st.session_state.rag_chain is None:
        if st.button("Setup AI Assistant"):
            with st.spinner("Setting up AI..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
                st.session_state.rag_chain = setup_rag_chain(groq_api_key, retriever)
                if st.session_state.rag_chain:
                    st.success("AI Assistant ready")

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.header("Disease Detection")
    
    uploaded_file = st.file_uploader("Choose cotton leaf image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("Analyze Disease"):
        if not uploaded_file:
            st.error("Upload image first")
        elif st.session_state.model is None:
            st.error("Load model first")
        else:
            with st.spinner("Analyzing..."):
                disease, confidence = predict_disease(st.session_state.model, image)
                
                if disease:
                    st.session_state.current_detection = create_detection_text(disease, confidence)
                    st.session_state.last_result = {'disease': disease, 'confidence': confidence}
                    st.success("Analysis complete")

with col2:
    st.header("Results")
    
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        st.write(f"Disease: {result['disease']}")
        st.write(f"Confidence: {result['confidence']:.1%}")
        
        # Show relevant treatment info
        disease_info = next((d for d in DISEASE_KNOWLEDGE if result['disease'] in d), None)
        if disease_info:
            lines = disease_info.split('\n')
            for line in lines:
                if line.strip():
                    st.write(line)
    else:
        st.info("Upload image and analyze to see results")

st.header("AI Assistant")

if not groq_api_key:
    st.warning("Enter Groq API key to use assistant")
elif not st.session_state.vectorstore:
    st.warning("Setup vector database first")
elif not st.session_state.rag_chain:
    st.warning("Setup AI assistant first")
else:
    # Show current detection context
    if 'current_detection' in st.session_state:
        st.info("Current Detection Available - Ask questions about this result")
    
    user_question = st.text_input("Ask about cotton diseases or treatments:")
    
    if user_question:
        with st.spinner("Getting answer..."):
            try:
                # Include current detection in context if available
                context_question = user_question
                if 'current_detection' in st.session_state:
                    context_question = f"{st.session_state.current_detection} Question: {user_question}"
                
                response = st.session_state.rag_chain.invoke(
                    {"input": context_question},
                    config={"configurable": {"session_id": "main_session"}}
                )
                
                st.write("Answer:")
                st.write(response["answer"])
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Show Chat History"):
            history = get_session_history("main_session")
            if history.messages:
                st.subheader("Chat History")
                for msg in history.messages[-6:]:  # Show last 6 messages
                    if hasattr(msg, 'content'):
                        role = "You" if msg.type == "human" else "Assistant"
                        st.write(f"{role}: {msg.content[:200]}...")
            else:
                st.info("No chat history")
    
    with col4:
        if st.button("Clear History"):
            if "main_session" in st.session_state.store:
                st.session_state.store["main_session"] = ChatMessageHistory()
                st.success("History cleared")
