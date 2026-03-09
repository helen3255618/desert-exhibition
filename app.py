import streamlit as st
import chromadb
from openai import OpenAI
import json
import base64
import io

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(
    page_title="Desert Exhibition",
    page_icon="🏜️",
    layout="centered"
)

st.title("🏜️ Desert Exhibition Assistant")
st.caption("Speak in Chinese, English, or French — I'll respond in your language.")

@st.cache_resource
def load_knowledge_base():
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection("exhibition")
    
    chunks = []
    for fname in ["desert_chunks_ch_001_054.jsonl", "desert_chunks_ch_055-100.jsonl", "desert_chunks_ch_101_130.jsonl"]:
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    
    texts = [c["text"] for c in chunks]
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    embeddings = [r.embedding for r in response.data]
    
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "chunk_id": c.get("chunk_id", ""),
            "domain": c.get("domain", ""),
            "exhibition_zone": c.get("exhibition_zone", ""),
            "subject": ", ".join(c.get("subject", [])) if isinstance(c.get("subject"), list) else "",
        } for c in chunks]
    )
    return collection

collection = load_knowledge_base()

SYSTEM_PROMPTS = {
    "zh": """你是一个博物馆展览的知识助手，风格是"诗意的科学家"：
语言有温度和画面感，但所有细节必须来自提供的来源，不添加、不虚构。
回答长度控制在80-100字之间，适合朗读。不要加来源标注。用中文回答。""",
    "en": """You are a museum exhibition assistant with the style of a poetic scientist.
Warm and vivid language, but every detail must come from the provided sources. Do not invent.
Keep responses to 80-100 words, suitable for reading aloud. No source references. Answer in English.""",
    "fr": """Tu es un assistant de musée au style de scientifique poétique.
Langage chaleureux, mais chaque détail doit provenir des sources. Ne pas inventer.
Réponse de 80-100 mots, adaptée à la lecture. Pas de références. Réponds en français."""
}

VOICES = {"zh": "nova", "en": "shimmer", "fr": "nova"}

def search(query, n=5):
    embedding = client.embeddings.create(
        input=query, model="text-embedding-3-small"
    ).data[0].embedding
    results = collection.query(query_embeddings=[embedding], n_results=n)
    return results["documents"][0], results["metadatas"][0]

def generate_answer(question, lang):
    docs, metas = search(question)
    context = "\n\n".join([f"[{i+1}] {d}" for i, d in enumerate(docs)])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[lang]},
            {"role": "user", "content": f"Content:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content, metas

def text_to_speech(text, lang):
    speech = client.audio.speech.create(
        model="tts-1", voice=VOICES[lang], input=text
    )
    return base64.b64encode(speech.content).decode()

if "history" not in st.session_state:
    st.session_state.history = []

# 录音
audio_input = st.audio_input("🎙️ Press to speak")

if audio_input is not None:
    with st.spinner("Processing..."):
        audio_bytes = audio_input.read()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )
        question = transcript.text
        lang_map = {"chinese": "zh", "english": "en", "french": "fr"}
        lang = lang_map.get(transcript.language, "en")

        answer, metas = generate_answer(question, lang)
        audio_b64 = text_to_speech(answer, lang)

        st.session_state.history.append({
            "question": question,
            "answer": answer,
            "audio": audio_b64,
            "lang": lang
        })

for item in reversed(st.session_state.history):
    st.markdown(f"**❓ {item['question']}**")
    st.markdown(f"💬 {item['answer']}")
    st.markdown(
        f'<audio autoplay controls src="data:audio/mp3;base64,{item["audio"]}"></audio>',
        unsafe_allow_html=True
    )
    st.divider()
