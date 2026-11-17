import streamlit as st
import requests
import time
import jiwer 
import pandas as pd
import sacrebleu
import difflib
import os

#python -m streamlit run app.py
#python -m uvicorn main:app --reload

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="YogadTranslate", layout="wide", initial_sidebar_state="collapsed")

# --- LOAD CORPUS DATASET ---
@st.cache_data
def load_corpus():
    try:
        return pd.read_csv("Yogad_Tourism_Corpus_upd - Sheet1.csv")
    except FileNotFoundError:
        st.error("Corpus file not found.")
        return pd.DataFrame()

corpus_df = load_corpus()

# --- CUSTOM HTML/CSS INJECTION ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Nunito:wght@400;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');

html, body, [data-testid="stAppViewContainer"], .main {
    scroll-behavior: smooth !important;
    background-color: #FFFFFF !important; 
    font-family: 'Nunito', sans-serif !important;
}

p, label, input, textarea, button, a, li { font-family: 'Nunito', sans-serif !important; }
h1, h2, h3, h4, h5, h6, .nav-brand, .metric-value { font-family: 'Lora', serif !important; }

/* Explicitly protect Streamlit's Material Icons */
.st-icon, span.material-symbols-rounded, span[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', sans-serif !important;
}
.st-visually-hidden {
    display: none !important;
}

/* Navbar */
.sticky-nav {
    position: fixed; top: 0; left: 0; width: 100%;
    background-color: #5A120F; border-bottom: 4px solid #E6B800; 
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 60px; z-index: 999999; box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
}
.nav-brand { color: #F8F5EE; font-size: 1.8rem; font-weight: 600; letter-spacing: 1px; }
.nav-links { display: flex; gap: 40px; }
.nav-links a { color: #F8F5EE; text-decoration: none; font-weight: 600; font-size: 1.1rem; transition: color 0.3s; }
.nav-links a:hover { color: #E6B800; }
.block-container { padding-top: 100px !important; max-width: 1200px !important; }

/* Selectboxes (Dark Mode Style) */
div[data-baseweb="select"] > div { 
    background-color: #262730 !important; 
    border: none !important;
    border-radius: 6px !important;
}
div[data-baseweb="select"] span { color: #FFFFFF !important; }
div[data-testid="stSelectbox"] label p { 
    color: #000000 !important; 
    font-size: 1rem !important; 
    font-weight: 700 !important; 
}

/* Text Areas (Cream Style) */
.stTextArea textarea { 
    background-color: #FCF9F0 !important; 
    border: 1px solid #CCCCCC !important; 
    border-radius: 8px !important; 
    color: #333333 !important; 
    font-size: 1.1rem !important; 
    padding: 15px !important; 
}
.stTextArea textarea:focus { border-color: #E6B800 !important; box-shadow: 0 0 0 1px #E6B800 !important; }
.stTextArea textarea:disabled { color: #333333 !important; -webkit-text-fill-color: #333333 !important; opacity: 1 !important; }

/* Secondary Buttons (Record, Play, Swap) */
button[data-testid="baseButton-secondary"] { 
    background-color: #262730 !important; 
    border: none !important; 
    color: #FFFFFF !important; 
    font-size: 1rem !important; 
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important; 
}
button[data-testid="baseButton-secondary"]:hover { background-color: #3f404d !important; color: #FFFFFF !important; }

/* Primary Button (Translate) */
button[data-testid="baseButton-primary"] { 
    background-color: #FF4B4B !important; 
    color: #FFFFFF !important; 
    border: none !important; 
    border-radius: 6px !important; 
    padding: 0.75rem 2rem !important; 
    font-size: 1.1rem !important; 
    font-weight: 700 !important; 
}
button[data-testid="baseButton-primary"]:hover { background-color: #ff3333 !important; }

div[data-testid="stAudioInput"] { background: #FFFFFF !important; border: 2px solid #CCCCCC !important; border-radius: 12px !important; padding: 10px !important; margin-bottom: 15px !important; }

/* Expander/Info */
div[data-testid="stAlert"] { background-color: #FCF9F0 !important; border-left: 4px solid #E6B800 !important; color: #333333 !important; border-radius: 8px !important; }
div[data-testid="stExpander"] { background-color: #FCF9F0 !important; border: 1px solid #E6B800 !important; border-radius: 8px !important; }
div[data-testid="stExpander"] summary { color: #5A120F !important; font-weight: 700 !important; font-size: 1rem !important; }
div[data-testid="stExpander"] summary svg { fill: #5A120F !important; }

/* Metrics */
.metric-box { background-color: #FCF9F0; border: 1px solid #E6B800; border-radius: 8px; padding: 15px; text-align: center; }
.metric-label { font-weight: 800; font-size: 1rem; color: #5A120F; margin-bottom: 5px; text-transform: uppercase; }

/* Info & Contact Sections */
.info-section, .contact-section { margin-top: 80px; }
.anchor { position: absolute; top: -120px; }
h2.section-header { color: #5A120F; font-size: 2.5rem; font-weight: 800; margin-bottom: 20px; }
</style>

<div class="sticky-nav">
    <div class="nav-brand">YogadTranslate</div>
    <div class="nav-links">
        <a href="#home">Home</a>
        <a href="#about">About</a>
        <a href="#contact">Contact</a>
    </div>
</div>
<div id="home" class="anchor" style="top: 0;"></div>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'translation_output' not in st.session_state: st.session_state.translation_output = ""
if 'base_translation_output' not in st.session_state: st.session_state.base_translation_output = ""
if 'show_metrics' not in st.session_state: st.session_state.show_metrics = False
if 'error_no_reference' not in st.session_state: st.session_state.error_no_reference = False
if 'suggested_correction' not in st.session_state: st.session_state.suggested_correction = None
if 'last_checked_text' not in st.session_state: st.session_state.last_checked_text = ""
if 'source_text_input' not in st.session_state: st.session_state.source_text_input = ""

if 'asr_notice' not in st.session_state: st.session_state.asr_notice = None
if 'last_processed_audio_hash' not in st.session_state: st.session_state.last_processed_audio_hash = None
if 'tts_audio_bytes' not in st.session_state: st.session_state.tts_audio_bytes = None
if 'tts_notice' not in st.session_state: st.session_state.tts_notice = None

if 'cer_score' not in st.session_state: st.session_state.cer_score = 0.0
if 'wer_score' not in st.session_state: st.session_state.wer_score = 0.0
if 'chrf_score' not in st.session_state: st.session_state.chrf_score = 0.0
if 'yt_speed' not in st.session_state: st.session_state.yt_speed = 0.0

if 'base_cer_score' not in st.session_state: st.session_state.base_cer_score = 0.0
if 'base_wer_score' not in st.session_state: st.session_state.base_wer_score = 0.0
if 'base_chrf_score' not in st.session_state: st.session_state.base_chrf_score = 0.0
if 'base_speed' not in st.session_state: st.session_state.base_speed = 0.0

if 'src_lang' not in st.session_state: st.session_state.src_lang = "Yogad"
if 'tgt_lang' not in st.session_state: st.session_state.tgt_lang = "Tagalog"
if 'show_audio_recorder' not in st.session_state: st.session_state.show_audio_recorder = False

def swap_languages():
    st.session_state.src_lang, st.session_state.tgt_lang = st.session_state.tgt_lang, st.session_state.src_lang
    st.session_state.translation_output = ""
    st.session_state.base_translation_output = ""
    st.session_state.show_metrics = False
    st.session_state.tts_audio_bytes = None
    st.session_state.tts_notice = None

def toggle_audio_recorder():
    st.session_state.show_audio_recorder = not st.session_state.show_audio_recorder

languages = ["Yogad", "Tagalog"]

def get_metric_color(metric_name, value):
    if metric_name in ["CER", "WER"]:
        if value <= 20: return "#28A745"
        elif value <= 40: return "#FD7E14"
        else: return "#DC3545"
    elif metric_name == "chrF++":
        if value >= 60: return "#28A745"
        elif value >= 40: return "#FD7E14"
        else: return "#DC3545"
    return "#333333"

def check_for_typos(input_text, source_language):
    if corpus_df is None or input_text.strip() == "": return None
    valid_words = set(" ".join(corpus_df[source_language].dropna().astype(str)).lower().split())
    input_words = input_text.lower().split()
    corrected_words = []
    typo_found = False
    
    for word in input_words:
        clean_word = ''.join(e for e in word if e.isalnum())
        if clean_word and clean_word not in valid_words:
            matches = difflib.get_close_matches(clean_word, valid_words, n=1, cutoff=0.8)
            if matches:
                corrected_words.append(word.replace(clean_word, matches[0]))
                typo_found = True
                continue
        corrected_words.append(word)
        
    if typo_found: return " ".join(corrected_words)
    return None

# ==========================================
# PART 1: THE TRANSLATION TOOL
# ==========================================
st.markdown("<div id='translation-tool'></div>", unsafe_allow_html=True)
st.write("") # Padding

col_from, col_swap, col_to = st.columns([4, 1, 4])
with col_from: st.selectbox("From", languages, key="src_lang")
with col_swap: 
    st.markdown("<div style='text-align: center; margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.button("⇄", type="secondary", on_click=swap_languages, use_container_width=True)
with col_to: st.selectbox("To", languages, key="tgt_lang")

col_src, col_tgt = st.columns(2)

with col_src:
    col_src_hdr, col_src_act = st.columns([3, 1], vertical_alignment="bottom")
    with col_src_hdr: 
        st.markdown(f"<div style='font-size: 1rem; color: #000000; font-weight: 700; margin-bottom: 5px; margin-top: 10px;'>Source Text ({st.session_state.src_lang})</div>", unsafe_allow_html=True)
    with col_src_act: 
        mic_label = "Hide Mic" if st.session_state.show_audio_recorder else "🎙️ Record"
        st.button(mic_label, key="toggle_mic_btn", on_click=toggle_audio_recorder, use_container_width=True)

    if st.session_state.show_audio_recorder:
        with st.container():
            st.markdown(f"<p style='font-size: 1rem; color: #5A120F; margin: 4px 0 8px 0; font-weight: 600;'>Click the microphone to speak in <b>{st.session_state.src_lang}</b>:</p>", unsafe_allow_html=True)
            audio_input = st.audio_input("Record Speech", key=f"audio_recorder_{st.session_state.src_lang}", label_visibility="collapsed")
            
            if audio_input is not None:
                audio_bytes = audio_input.getvalue()
                audio_hash = hash(audio_bytes)
                
                if st.session_state.last_processed_audio_hash != audio_hash:
                    st.session_state.last_processed_audio_hash = audio_hash
                    with st.spinner(f"Transcribing {st.session_state.src_lang} audio via Wav2Vec 2.0..."):
                        try:
                            files = {"audio_file": ("recording.wav", audio_bytes, "audio/wav")}
                            data = {"src_lang": st.session_state.src_lang}
                            asr_res = requests.post("http://127.0.0.1:8000/transcribe", files=files, data=data, timeout=120)
                            
                            if asr_res.status_code == 200:
                                res_payload = asr_res.json()
                                
                                if res_payload.get("status") == "success":
                                    raw_transcription = res_payload.get("transcription", "")
                                    duration = res_payload.get("duration", 0.0)
                                    
                                    auto_corrected = check_for_typos(raw_transcription, st.session_state.src_lang)
                                    
                                    if auto_corrected and auto_corrected != raw_transcription:
                                        st.session_state.source_text_input = auto_corrected
                                        st.session_state.asr_notice = f"Transcribed {duration}s: \"{raw_transcription}\" ➔ Auto-corrected to: \"{auto_corrected}\""
                                    else:
                                        st.session_state.source_text_input = raw_transcription
                                        st.session_state.asr_notice = f"Transcribed {duration}s of {st.session_state.src_lang}: \"{raw_transcription}\""
                                        
                                    st.rerun()
                                    
                                elif res_payload.get("status") == "pending_model":
                                    st.info(f"ℹ️ {res_payload.get('message')}")
                            else:
                                st.error("Speech recognition service returned an error.")
                        except requests.exceptions.ConnectionError:
                            st.error("Backend Connection Failed. Make sure your FastAPI server is running.")

    if st.session_state.asr_notice:
        st.success(st.session_state.asr_notice)
        st.session_state.asr_notice = None
        
    source_text = st.text_area("Source Text", key="source_text_input", placeholder=f"Type {st.session_state.src_lang} text here... (Max 250 characters)", height=200, max_chars=250, label_visibility="collapsed")
    st.caption(f"**Word Count:** {len(source_text.split())}")
        
        # --- NEW: CUSTOM REFERENCE BOX ---
    st.markdown(f"<div style='font-size: 1rem; color: #5A120F; font-weight: 800; margin-top: 15px;'>Custom Reference (Optional)</div>", unsafe_allow_html=True)
    custom_reference = st.text_input(f"Provide the exact {st.session_state.tgt_lang} translation to calculate metrics for novel phrases.", key="custom_ref")

with col_tgt:
    col_tgt_hdr, col_tgt_act = st.columns([3, 1], vertical_alignment="bottom")
    with col_tgt_hdr: 
        st.markdown(f"<div style='font-size: 1rem; color: #000000; font-weight: 700; margin-bottom: 5px; margin-top: 10px;'>Translation Output ({st.session_state.tgt_lang})</div>", unsafe_allow_html=True)
    with col_tgt_act: 
        if st.button("🔊 Play", key="tts_btn", use_container_width=True):
            if st.session_state.translation_output:
                with st.spinner(f"Synthesizing {st.session_state.tgt_lang} speech via VITS..."):
                    try:
                        tts_res = requests.post(
                            "http://127.0.0.1:8000/synthesize",
                            json={"text": st.session_state.translation_output, "lang": st.session_state.tgt_lang},
                            timeout=120
                        )
                        if tts_res.status_code == 200:
                            st.session_state.tts_audio_bytes = tts_res.content
                            st.session_state.tts_notice = None
                        elif tts_res.status_code == 503:
                            err_data = tts_res.json()
                            st.session_state.tts_audio_bytes = None
                            st.session_state.tts_notice = err_data.get("message", "Model pending training.")
                        else:
                            st.error("Failed to generate speech synthesis.")
                    except requests.exceptions.ConnectionError:
                        st.error("Backend Connection Failed. Please ensure FastAPI is running.")
            else:
                st.toast("Please translate text first before generating audio playback.")
        
    output_val = st.session_state.translation_output if st.session_state.translation_output else ""
    st.text_area("Translation", value=output_val, height=200, label_visibility="collapsed", disabled=True)
    
    if st.session_state.tts_audio_bytes:
        st.audio(st.session_state.tts_audio_bytes, format="audio/wav", autoplay=True)
    elif st.session_state.tts_notice:
        st.info(f"ℹ️ {st.session_state.tts_notice}")

if st.session_state.suggested_correction and st.session_state.last_checked_text == source_text:
    st.markdown(f"<div style='background-color: #FCF9F0; border-left: 4px solid #E6B800; padding: 15px; color: #333333; border-radius: 8px; font-size: 1.1rem; margin-bottom: 15px;'> Did you mean: <b>{st.session_state.suggested_correction}</b>?</div>", unsafe_allow_html=True)
    
    col_yes, col_no, col_space = st.columns([1, 1, 2])
    with col_yes:
        if st.button("Yes, use correction"):
            st.session_state.source_text_input = st.session_state.suggested_correction
            st.session_state.suggested_correction = None
            st.session_state.force_translate = True 
            st.rerun()
    with col_no:
        if st.button("No, translate as typed"):
            st.session_state.suggested_correction = None
            st.session_state.force_translate = True 
            st.rerun()

st.write("")
col_empty1, col_btn, col_empty2 = st.columns([1, 2, 1])

with col_btn:
    if st.button("Translate & Evaluate", type="primary", use_container_width=True) or st.session_state.get('force_translate', False):
        st.session_state.force_translate = False 
        
        if source_text.strip() != "":
            possible_correction = check_for_typos(source_text, st.session_state.src_lang)
            
            if possible_correction and st.session_state.get('last_checked_text') != source_text:
                st.session_state.last_checked_text = source_text
                st.session_state.suggested_correction = possible_correction
                st.rerun()
            else:
                st.session_state.suggested_correction = None
                st.session_state.last_checked_text = source_text
                
                with st.spinner("Translating via AI Models..."):
                    try:
                        start_time_yt = time.time()
                        response_finetuned = requests.post(
                            "http://127.0.0.1:8000/translate",
                            json={"text": source_text, "src_lang": st.session_state.src_lang, "tgt_lang": st.session_state.tgt_lang},
                            timeout=120
                        )
                        yt_duration = time.time() - start_time_yt
                        
                        start_time_base = time.time()
                        response_base = requests.post(
                            "http://127.0.0.1:8000/translate_base",
                            json={"text": source_text, "src_lang": st.session_state.src_lang, "tgt_lang": st.session_state.tgt_lang},
                            timeout=120
                        )
                        base_duration = time.time() - start_time_base
                        
                        if response_finetuned.status_code == 200 and response_base.status_code == 200:
                            predicted_output = response_finetuned.json()["translation"]
                            base_output = response_base.json()["translation"]
                            
                            st.session_state.translation_output = predicted_output
                            st.session_state.base_translation_output = base_output
                            st.session_state.yt_speed = round(yt_duration, 2)
                            st.session_state.base_speed = round(base_duration, 2)
                            st.session_state.tts_audio_bytes = None
                            st.session_state.tts_notice = None

                            reference_text = None
                                
                                # 1. First, check if the user provided a manual ground-truth reference
                            if st.session_state.custom_ref and st.session_state.custom_ref.strip() != "":
                                    reference_text = st.session_state.custom_ref.strip()
                                    
                                # 2. If not, try to find it in the CSV dataset
                            elif corpus_df is not None:
                                    src_col = st.session_state.src_lang
                                    tgt_col = st.session_state.tgt_lang
                                    if src_col in corpus_df.columns and tgt_col in corpus_df.columns:
                                        match = corpus_df[corpus_df[src_col].astype(str).str.strip().str.lower() == source_text.strip().lower()]
                                        if not match.empty:
                                            reference_text = str(match.iloc[0][tgt_col])
                            
                            if reference_text:
                                st.session_state.cer_score = round(jiwer.cer(reference_text.strip(), predicted_output) * 100, 2)
                                st.session_state.wer_score = round(jiwer.wer(reference_text.strip(), predicted_output) * 100, 2)
                                st.session_state.chrf_score = round(sacrebleu.corpus_chrf([predicted_output], [[reference_text.strip()]], word_order=2).score, 2)

                                st.session_state.base_cer_score = round(jiwer.cer(reference_text.strip(), base_output) * 100, 2)
                                st.session_state.base_wer_score = round(jiwer.wer(reference_text.strip(), base_output) * 100, 2)
                                st.session_state.base_chrf_score = round(sacrebleu.corpus_chrf([base_output], [[reference_text.strip()]], word_order=2).score, 2)

                                st.session_state.show_metrics = True
                                st.session_state.error_no_reference = False
                            else:
                                st.session_state.show_metrics = False
                                st.session_state.error_no_reference = True

                            st.rerun()
                        else:
                            st.error(f"Finetuned Model: {response_finetuned.text} | Base Model: {response_base.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("Connection Failed! Make sure your Uvicorn backend is running.")
        else:
            st.warning("Please enter text or record speech to translate.")

st.write("---")
with st.expander(" Suggest an Edit or Add New Word"):
    st.markdown("Help us improve the dictionary! Select an existing word to suggest a correction, or add a completely new one.")
    if corpus_df is not None and not corpus_df.empty:
        existing_words = sorted(list(set(corpus_df[st.session_state.src_lang].dropna().astype(str).tolist())))
    else:
        existing_words = []
        
    fallback_option = " Not found? Add a new word instead"
    dropdown_options = [fallback_option] + existing_words
    
    selected_option = st.selectbox(f"Search existing {st.session_state.src_lang} word:", dropdown_options)
    
    if selected_option == fallback_option:
        st.info(" Adding a completely new word.")
        s_col1, s_col2 = st.columns(2)
        with s_col1: sugg_source = st.text_input(f"New {st.session_state.src_lang} Word/Phrase")
        with s_col2: sugg_target = st.text_input(f"{st.session_state.tgt_lang} Translation")
        action_type = "add_new"
    else:
        st.info(f"Suggesting an edit for: **{selected_option}**")
        s_col1, s_col2 = st.columns(2)
        with s_col1: 
            sugg_source = st.text_input(f"Correction for {st.session_state.src_lang}", value=selected_option)
        with s_col2: 
            current_trans = corpus_df[corpus_df[st.session_state.src_lang] == selected_option][st.session_state.tgt_lang].iloc[0]
            sugg_target = st.text_input(f"Better translation for {st.session_state.tgt_lang}", value=current_trans)
        action_type = "edit_existing"
        
    if st.button("Submit Suggestion For Approval"):
        if sugg_source and sugg_target:
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/submit_suggestion",
                    json={
                        "src_lang": st.session_state.src_lang,
                        "src_text": sugg_source,
                        "tgt_lang": st.session_state.tgt_lang,
                        "tgt_text": sugg_target,
                        "action_type": action_type
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    st.success("Thank you! Your suggestion has been sent to the developers for review and approval.")
                else:
                    st.error("Failed to submit. Backend returned an error.")
            except requests.exceptions.ConnectionError:
                st.error("Connection Failed! Make sure your Uvicorn backend is running.")
        else:
            st.error("Please fill in both fields.")

st.write("<br><br>", unsafe_allow_html=True)

# ==========================================
# PART 2: TRANSLATION EVALUATION SECTION
# ==========================================
st.markdown("<div id='evaluation'></div>", unsafe_allow_html=True)
st.markdown("<h2 style='color: #5A120F; font-weight: 700; border-bottom: 2px solid #E6B800; padding-bottom: 5px; margin-top: 40px;'>Translation Evaluation</h2>", unsafe_allow_html=True)

with st.expander("Evaluation Metrics Legend"):
    st.info("""
    **📊 Evaluation Metrics Legend:**
    *   **CER (Character Error Rate) & WER (Word Error Rate):** **LOWER is better.** (🟢 **0-20%:** Excellent | 🟠 **21-40%:** Acceptable | 🔴 **>40%:** Poor)
    *   **chrF++ (Character n-gram F-score):** **HIGHER is better.** (🟢 **>60:** Excellent | 🟠 **40-60:** Acceptable | 🔴 **<40:** Poor)
    """)

if st.session_state.show_metrics:
    col_baseline, col_spacer, col_proposed = st.columns([1, 0.05, 1])

    with col_baseline:
        st.markdown("<h3 style='text-align: center; color: #5A120F; font-weight: 700; margin-bottom: 10px;'>Base NLLB-200 Model<br><span style='font-size: 1rem; color: #B28D00;'>(Pre-Trained / Baseline)</span></h3>", unsafe_allow_html=True)
        st.markdown("<p style='margin-bottom: 5px; font-weight: 700; color: #333;'>Model Output:</p>", unsafe_allow_html=True)
        st.text_area("Base Output", value=st.session_state.base_translation_output, height=100, label_visibility="collapsed", disabled=True)

        st.write("")
        b1, b2, b3, b4 = st.columns(4)
        with b1: st.markdown(f"<div class='metric-box'><div class='metric-label'>CER</div><div style='font-size: 1.5rem; font-weight: 800; color: {get_metric_color('CER', st.session_state.base_cer_score)};'>{st.session_state.base_cer_score}%</div></div>", unsafe_allow_html=True)
        with b2: st.markdown(f"<div class='metric-box'><div class='metric-label'>WER</div><div style='font-size: 1.5rem; font-weight: 800; color: {get_metric_color('WER', st.session_state.base_wer_score)};'>{st.session_state.base_wer_score}%</div></div>", unsafe_allow_html=True)
        with b3: st.markdown(f"<div class='metric-box'><div class='metric-label'>chrF++</div><div style='font-size: 1.5rem; font-weight: 800; color: {get_metric_color('chrF++', st.session_state.base_chrf_score)};'>{st.session_state.base_chrf_score}</div></div>", unsafe_allow_html=True)
        with b4: st.markdown(f"<div class='metric-box'><div class='metric-label'>Speed</div><div style='font-size: 1.5rem; font-weight: 800; color: #333333;'>{st.session_state.base_speed}s</div></div>", unsafe_allow_html=True)

    with col_proposed:
        st.markdown("<h3 style='text-align: center; color: #5A120F; font-weight: 700; margin-bottom: 10px;'>Proposed System<br><span style='font-size: 1rem; color: #B28D00;'>(Fine-Tuned Model)</span></h3>", unsafe_allow_html=True)
        st.markdown("<p style='margin-bottom: 5px; font-weight: 700; color: #333;'>Model Output:</p>", unsafe_allow_html=True)
        st.text_area("YT Output", value=st.session_state.translation_output, height=100, label_visibility="collapsed", disabled=True)
        
        st.write("")
        p1, p2, p3, p4 = st.columns(4)
        with p1: st.markdown(f"<div class='metric-box'><div class='metric-label'>CER</div><div style='font-size: 1.5rem; font-weight: 800; color: {get_metric_color('CER', st.session_state.cer_score)};'>{st.session_state.cer_score}%</div></div>", unsafe_allow_html=True)
        with p2: st.markdown(f"<div class='metric-box'><div class='metric-label'>WER</div><div style='font-size: 1.5rem; font-weight: 800; color: {get_metric_color('WER', st.session_state.wer_score)};'>{st.session_state.wer_score}%</div></div>", unsafe_allow_html=True)
        with p3: st.markdown(f"<div class='metric-box'><div class='metric-label'>chrF++</div><div style='font-size: 1.5rem; font-weight: 800; color: {get_metric_color('chrF++', st.session_state.chrf_score)};'>{st.session_state.chrf_score}</div></div>", unsafe_allow_html=True)
        with p4: st.markdown(f"<div class='metric-box'><div class='metric-label'>Speed</div><div style='font-size: 1.5rem; font-weight: 800; color: #333333;'>{st.session_state.yt_speed}s</div></div>", unsafe_allow_html=True)

elif st.session_state.error_no_reference:
    st.warning("Phrase not found in exact dataset. AI successfully translated the novel sentence, but evaluation metrics require a verified human reference to calculate.")
else:
    st.markdown("<p style='text-align: left; color: #666; font-size: 1.1rem; padding: 20px 0;'>Run a translation from the tool above to generate evaluation metrics.</p>", unsafe_allow_html=True)

st.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True)

# ==========================================
# PART 3: ABOUT SECTION
# ==========================================
st.markdown("""
<div class="info-section">
<div id="about" class="anchor"></div>
<h2 class="section-header">About the Project</h2>
<p style="font-size: 1.2rem; line-height: 1.6; color: #333; margin-bottom: 30px;">This system was developed as part of an academic thesis dedicated to the digital preservation of the Yogad language in the Philippines. It serves as an end-to-end multimodal pipeline integrating state-of-the-art models like Wav2Vec 2.0 for automatic speech recognition, NLLB-200 for accurate text translation, and VITS for speech synthesis. By bridging Yogad and Tagalog, this tool aims to provide a reliable platform for linguistic preservation and accessibility.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)

# ==========================================
# PART 4: CONTACT SECTION
# ==========================================
st.markdown("""
<div class="contact-section">
<div id="contact" class="anchor"></div>
<h2 class="section-header">The Developers</h2>
<div style="display: flex; gap: 40px; margin-top: 20px;">
    <div style="background-color: #FCF9F0; border: 1px solid #E6B800; border-radius: 12px; padding: 30px; width: 50%;">
        <h4 style="color: #5A120F; font-size: 1.5rem; margin-bottom: 5px; font-weight: 700;">Jan Bertson C. Ambatali</h4>
        <h6 style="color: #666; font-size: 1.1rem; margin-bottom: 15px;">BSCS 3-1 Researcher</h6>
        <p style="color: #333; margin-bottom: 5px;">Isabela State University Echague campus</p>
        <a href="mailto:janbertson.c.ambatali@isu.edu.ph" style="color: #5A120F; font-weight: 600; text-decoration: none;">janbertson.c.ambatali@isu.edu.ph</a>
    </div>
    <div style="background-color: #FCF9F0; border: 1px solid #E6B800; border-radius: 12px; padding: 30px; width: 50%;">
        <h4 style="color: #5A120F; font-size: 1.5rem; margin-bottom: 5px; font-weight: 700;">Christian Jay E. San jose</h4>
        <h6 style="color: #666; font-size: 1.1rem; margin-bottom: 15px;">BSCS 3-1 Researcher</h6>
        <p style="color: #333; margin-bottom: 5px;">Isabela State University Echague campus</p>
        <a href="mailto:christianjay.e.sanjose@isu.edu.ph" style="color: #5A120F; font-weight: 600; text-decoration: none;">christianjay.e.sanjose@isu.edu.ph</a>
    </div>
</div>
</div>
""", unsafe_allow_html=True)