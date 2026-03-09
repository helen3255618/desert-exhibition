{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import chromadb\
from openai import OpenAI\
import json\
import base64\
import io\
from audiorecorder import audiorecorder\
\
# \uc0\u21021 \u22987 \u21270 \
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])\
\
# \uc0\u39029 \u38754 \u35774 \u32622 \
st.set_page_config(\
    page_title="\uc0\u55356 \u57308 \u65039  Desert Exhibition",\
    page_icon="\uc0\u55356 \u57308 \u65039 ",\
    layout="centered"\
)\
\
st.title("\uc0\u55356 \u57308 \u65039  Desert Exhibition Assistant")\
st.caption("Speak in Chinese, English, or French \'97 I'll respond in your language.")\
\
# \uc0\u21152 \u36733 chunks\u24182 \u24314 \u21521 \u37327 \u24211 \u65288 \u21482 \u22312 \u31532 \u19968 \u27425 \u36816 \u34892 \u26102 \u25191 \u34892 \u65289 \
@st.cache_resource\
def load_knowledge_base():\
    chroma_client = chromadb.Client()\
    collection = chroma_client.create_collection("exhibition")\
    \
    chunks = []\
    for fname in ["desert_chunks_ch_001_054.jsonl", "desert_chunks_ch_055-100.jsonl", "desert_chunks_ch_101_130.jsonl"]:  # \uc0\u20320 \u30340 \u25991 \u20214 \u21517 \
        with open(fname, "r", encoding="utf-8") as f:\
            for line in f:\
                line = line.strip()\
                if line:\
                    chunks.append(json.loads(line))\
    \
    # \uc0\u25209 \u37327 \u29983 \u25104 embedding\
    texts = [c["text"] for c in chunks]\
    response = client.embeddings.create(\
        input=texts,\
        model="text-embedding-3-small"\
    )\
    embeddings = [r.embedding for r in response.data]\
    \
    collection.add(\
        ids=[c["chunk_id"] for c in chunks],\
        embeddings=embeddings,\
        documents=texts,\
        metadatas=[\{\
            "chunk_id": c.get("chunk_id", ""),\
            "domain": c.get("domain", ""),\
            "exhibition_zone": c.get("exhibition_zone", ""),\
            "subject": ", ".join(c.get("subject", [])),\
        \} for c in chunks]\
    )\
    return collection\
\
collection = load_knowledge_base()\
\
# \uc0\u31995 \u32479 \u25552 \u31034 \
SYSTEM_PROMPTS = \{\
    "zh": """\uc0\u20320 \u26159 \u19968 \u20010 \u21338 \u29289 \u39302 \u23637 \u35272 \u30340 \u30693 \u35782 \u21161 \u25163 \u65292 \u39118 \u26684 \u26159 "\u35799 \u24847 \u30340 \u31185 \u23398 \u23478 "\u65306 \
\uc0\u35821 \u35328 \u26377 \u28201 \u24230 \u21644 \u30011 \u38754 \u24863 \u65292 \u20294 \u25152 \u26377 \u32454 \u33410 \u24517 \u39035 \u26469 \u33258 \u25552 \u20379 \u30340 \u26469 \u28304 \u65292 \u19981 \u28155 \u21152 \u12289 \u19981 \u34394 \u26500 \u12290 \
\uc0\u22238 \u31572 \u38271 \u24230 \u25511 \u21046 \u22312 80-100\u23383 \u20043 \u38388 \u65292 \u36866 \u21512 \u26391 \u35835 \u12290 \u19981 \u35201 \u21152 \u26469 \u28304 \u26631 \u27880 \u12290 \u29992 \u20013 \u25991 \u22238 \u31572 \u12290 """,\
    "en": """You are a museum exhibition assistant with the style of a "poetic scientist":\
Warm and vivid language, but every detail must come from the provided sources. Do not invent.\
Keep responses to 80-100 words, suitable for reading aloud. No source references. Answer in English.""",\
    "fr": """Tu es un assistant de mus\'e9e au style de "scientifique po\'e9tique":\
Langage chaleureux, mais chaque d\'e9tail doit provenir des sources. Ne pas inventer.\
R\'e9ponse de 80-100 mots, adapt\'e9e \'e0 la lecture. Pas de r\'e9f\'e9rences. R\'e9ponds en fran\'e7ais."""\
\}\
\
VOICES = \{"zh": "nova", "en": "shimmer", "fr": "nova"\}\
\
def search(query, n=5):\
    embedding = client.embeddings.create(\
        input=query, model="text-embedding-3-small"\
    ).data[0].embedding\
    results = collection.query(query_embeddings=[embedding], n_results=n)\
    return results["documents"][0], results["metadatas"][0]\
\
def generate_answer(question, lang):\
    docs, metas = search(question)\
    context = "\\n\\n".join([f"[\{i+1\}] \{d\}" for i, d in enumerate(docs)])\
    response = client.chat.completions.create(\
        model="gpt-4o-mini",\
        messages=[\
            \{"role": "system", "content": SYSTEM_PROMPTS[lang]\},\
            \{"role": "user", "content": f"Content:\\n\{context\}\\n\\nQuestion: \{question\}"\}\
        ]\
    )\
    return response.choices[0].message.content, metas\
\
def text_to_speech(text, lang):\
    speech = client.audio.speech.create(\
        model="tts-1", voice=VOICES[lang], input=text\
    )\
    return base64.b64encode(speech.content).decode()\
\
# \uc0\u23545 \u35805 \u21382 \u21490 \
if "history" not in st.session_state:\
    st.session_state.history = []\
\
# \uc0\u24405 \u38899 \u30028 \u38754 \
audio = audiorecorder("\uc0\u55356 \u57241 \u65039  Press to speak", "\u9209 \u65039  Recording... press to stop")\
\
if len(audio) > 0:\
    # \uc0\u36716 \u25104 bytes\
    audio_bytes = io.BytesIO()\
    audio.export(audio_bytes, format="wav")\
    audio_bytes.seek(0)\
\
    with st.spinner("Processing..."):\
        # Whisper\uc0\u35782 \u21035 \
        audio_bytes.name = "recording.wav"\
        transcript = client.audio.transcriptions.create(\
            model="whisper-1",\
            file=audio_bytes,\
            response_format="verbose_json"\
        )\
        question = transcript.text\
        lang_map = \{"chinese": "zh", "english": "en", "french": "fr"\}\
        lang = lang_map.get(transcript.language, "en")\
\
        # \uc0\u29983 \u25104 \u22238 \u31572 \
        answer, metas = generate_answer(question, lang)\
\
        # TTS\
        audio_b64 = text_to_speech(answer, lang)\
\
        # \uc0\u23384 \u20837 \u21382 \u21490 \
        st.session_state.history.append(\{\
            "question": question,\
            "answer": answer,\
            "audio": audio_b64,\
            "lang": lang\
        \})\
\
# \uc0\u26174 \u31034 \u23545 \u35805 \u21382 \u21490 \
for item in reversed(st.session_state.history):\
    st.markdown(f"**\uc0\u10067  \{item['question']\}**")\
    st.markdown(f"\uc0\u55357 \u56492  \{item['answer']\}")\
    st.markdown(\
        f'<audio autoplay controls src="data:audio/mp3;base64,\{item["audio"]\}"></audio>',\
        unsafe_allow_html=True\
    )\
    st.divider()\
```\
\
---\
\
## \uc0\u31532 \u20108 \u27493 \u65306 \u26032 \u24314 requirements.txt\
\
\uc0\u22312 \u21516 \u19968 \u20010 \u25991 \u20214 \u22841 \u37324 \u65292 \u26032 \u24314 \u19968 \u20010 \u25991 \u20214 \u21483  `requirements.txt`\u65292 \u20869 \u23481 \u22914 \u19979 \u65306 \
```\
openai\
chromadb\
streamlit\
audiorecorder}