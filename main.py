import os
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

import sys
import glob
import csv
import io
import re
import unicodedata
import asyncio
import gc
import traceback
import numpy as np
import soundfile as sf
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AutoProcessor, 
    Wav2Vec2ForCTC, 
    VitsModel
)

# Run instructions:
# python -m uvicorn main:app --reload
# python -m streamlit run app.py

# --- 1. INITIALIZE API & DIRECTORIES ---
app = FastAPI(title="Yogad Translate AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
YOGAD_TTS_DIR = os.path.join(current_dir, "Yogad_VITS_Model")

# Add Yogad_VITS_Model directory to sys.path so native VITS modules can be imported
if os.path.exists(YOGAD_TTS_DIR) and YOGAD_TTS_DIR not in sys.path:
    sys.path.append(YOGAD_TTS_DIR)

# Safe imports for native VITS modules
try:
    import utils
    import commons
    from models import SynthesizerTrn
    from text import text_to_sequence
    from text.symbols import symbols
    NATIVE_VITS_AVAILABLE = True
except ImportError as e:
    print(f" Native VITS modules not fully found in {YOGAD_TTS_DIR}: {e}")
    NATIVE_VITS_AVAILABLE = False

gpu_lock = asyncio.Lock()
device = "cuda" if torch.cuda.is_available() else "cpu"

def flush_vram():
    """Forces PyTorch to release unused VRAM back to the operating system."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# --- 2. TEXT NORMALIZER ---
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# --- 3. LOAD BI-DIRECTIONAL HYBRID DICTIONARY ---
print("Loading Bi-Directional Hybrid Dictionary...")
try:
    df = pd.read_csv("Yogad_Tourism_Corpus_upd - Sheet1.csv")
    df['yogad_search'] = df['Yogad'].apply(normalize_text)
    df['tgl_search'] = df['Tagalog'].apply(normalize_text)
    
    tgl_to_yogad_dict = dict(zip(df['tgl_search'], df['Yogad'].astype(str)))
    yogad_to_tgl_dict = dict(zip(df['yogad_search'], df['Tagalog'].astype(str)))
    print(f"Successfully loaded {len(yogad_to_tgl_dict)} hybrid dictionary pairs!")
except Exception as e:
    print(f"WARNING: Could not load CSV. Hybrid intercepts disabled. Error: {e}")
    tgl_to_yogad_dict = {}
    yogad_to_tgl_dict = {}

# --- 4. LOAD FULL FINE-TUNED NLLB TRANSLATION MODEL ---
print("Loading Fine-Tuned Yogad NLLB Model...")
model_path = os.path.join(current_dir, "Yogad_NLLB_Final")

tokenizer = None
model = None
base_model = None

if not os.path.exists(model_path):
    print(" CRITICAL ERROR: The Yogad_NLLB_Final folder is missing!")
else:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            src_lang="tgl_Latn", 
            tgt_lang="ilo_Latn"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model = model.to(device)
        model.eval()
        print(f" Fine-Tuned NLLB Model Successfully Loaded on {device.upper()}!")
    except Exception as e:
        print(f" CRITICAL ERROR LOADING NLLB MODEL: {e}")

def get_base_nllb_model():
    """Lazy loader for baseline pre-trained NLLB model for comparative evaluation."""
    global base_model
    if base_model is None:
        print("Loading baseline facebook/nllb-200-distilled-600M for comparison...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        base_model.eval()
    return base_model

# --- 5. WAV2VEC 2.0 SPEECH RECOGNITION (ASR) CONFIGURATION ---
TAGALOG_ASR_REPO = "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official"
YOGAD_ASR_DIR = os.path.join(current_dir, "Yogad_Wav2Vec2_Final")

asr_registry = {
    "Tagalog": {"processor": None, "model": None},
    "Yogad": {"processor": None, "model": None}
}

def get_asr_engine(lang: str):
    if lang == "Tagalog":
        if asr_registry["Tagalog"]["model"] is None:
            try:
                proc = AutoProcessor.from_pretrained(TAGALOG_ASR_REPO)
                asr_mod = Wav2Vec2ForCTC.from_pretrained(TAGALOG_ASR_REPO).to(device)
                asr_mod.eval()
                asr_registry["Tagalog"]["processor"] = proc
                asr_registry["Tagalog"]["model"] = asr_mod
            except Exception as e:
                return None, None, f"Tagalog ASR model load failed: {e}"
        return asr_registry["Tagalog"]["processor"], asr_registry["Tagalog"]["model"], None

    elif lang == "Yogad":
        if asr_registry["Yogad"]["model"] is None:
            if not os.path.exists(YOGAD_ASR_DIR):
                return None, None, "Yogad Wav2Vec 2.0 model weights pending."
            try:
                proc = AutoProcessor.from_pretrained(YOGAD_ASR_DIR)
                asr_mod = Wav2Vec2ForCTC.from_pretrained(YOGAD_ASR_DIR).to(device)
                asr_mod.eval()
                asr_registry["Yogad"]["processor"] = proc
                asr_registry["Yogad"]["model"] = asr_mod
            except Exception as e:
                return None, None, f"Yogad ASR loading error: {e}"
        return asr_registry["Yogad"]["processor"], asr_registry["Yogad"]["model"], None

    return None, None, f"Unsupported speech language '{lang}'."

# --- 6. VITS TEXT-TO-SPEECH (TTS) CONFIGURATION ---
TAGALOG_TTS_REPO = "facebook/mms-tts-tgl"

vits_registry = {
    "Tagalog": {"tokenizer": None, "model": None},
    "Yogad": {"hps": None, "model": None}
}

def get_yogad_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)

def get_vits_engine(lang: str):
    if lang == "Tagalog":
        if vits_registry["Tagalog"]["model"] is None:
            try:
                tok = AutoTokenizer.from_pretrained(TAGALOG_TTS_REPO)
                vits_mod = VitsModel.from_pretrained(TAGALOG_TTS_REPO).to(device)
                vits_mod.eval()
                vits_registry["Tagalog"]["tokenizer"] = tok
                vits_registry["Tagalog"]["model"] = vits_mod
            except Exception as e:
                return None, None, f"Tagalog VITS load failed: {e}"
        return vits_registry["Tagalog"]["tokenizer"], vits_registry["Tagalog"]["model"], None

    elif lang == "Yogad":
        if not NATIVE_VITS_AVAILABLE:
            return None, None, "Native VITS modules (models.py, utils.py) not available in Yogad_VITS_Model."
            
        if vits_registry["Yogad"]["model"] is None:
            config_candidates = [
                os.path.join(YOGAD_TTS_DIR, "f1_config.json"),
                os.path.join(YOGAD_TTS_DIR, "config.json")
            ]
            config_path = next((c for c in config_candidates if os.path.exists(c)), None)
            
            pth_candidates = sorted(glob.glob(os.path.join(YOGAD_TTS_DIR, "G_*.pth")), key=os.path.getmtime)
            model_path_tts = pth_candidates[-1] if pth_candidates else None
            
            if not config_path:
                return None, None, "Missing config.json or f1_config.json in Yogad_VITS_Model folder."
            if not model_path_tts:
                return None, None, "Missing generator checkpoint (G_*.pth) in Yogad_VITS_Model folder."
            
            try:
                hps = utils.get_hparams_from_file(config_path)
                net_g = SynthesizerTrn(
                    len(symbols),
                    hps.data.filter_length // 2 + 1,
                    hps.train.segment_size // hps.data.hop_length,
                    **hps.model
                ).to(device)
                net_g.eval()
                utils.load_checkpoint(model_path_tts, net_g, None)
                
                vits_registry["Yogad"]["hps"] = hps
                vits_registry["Yogad"]["model"] = net_g
                print(f" Loaded native Yogad VITS checkpoint: {os.path.basename(model_path_tts)}")
            except Exception as e:
                return None, None, f"Yogad VITS loading error: {e}"
                
        return vits_registry["Yogad"]["hps"], vits_registry["Yogad"]["model"], None

    return None, None, f"Unsupported synthesis language '{lang}'."

# --- 7. DATA SCHEMAS ---
class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

class TTSRequest(BaseModel):
    text: str
    lang: str = "Tagalog"

class SuggestionRequest(BaseModel):
    src_lang: str
    src_text: str
    tgt_lang: str
    tgt_text: str
    action_type: str

# --- 8. TRANSLATION ENDPOINTS ---
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    input_text = request.text.strip()
    if not input_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    search_query = normalize_text(input_text)
    
    # 1. Full Sentence Exact Match
    if request.src_lang == "Tagalog" and request.tgt_lang == "Yogad":
        if search_query in tgl_to_yogad_dict:
            return {"original": input_text, "translation": tgl_to_yogad_dict[search_query]}

    elif request.src_lang == "Yogad" and request.tgt_lang == "Tagalog":
        if search_query in yogad_to_tgl_dict:
            return {"original": input_text, "translation": yogad_to_tgl_dict[search_query]}
            
    # 2. Word-by-Word Combination Fallback
    words = input_text.split()
    fallback_translation = None
    
    if len(words) > 1:
        translated_words = []
        all_words_found = True
        
        for w in words:
            clean_w = normalize_text(w)
            if request.src_lang == "Yogad" and request.tgt_lang == "Tagalog":
                if clean_w in yogad_to_tgl_dict:
                    translated_words.append(yogad_to_tgl_dict[clean_w])
                else:
                    all_words_found = False
                    break
            elif request.src_lang == "Tagalog" and request.tgt_lang == "Yogad":
                if clean_w in tgl_to_yogad_dict:
                    translated_words.append(tgl_to_yogad_dict[clean_w])
                else:
                    all_words_found = False
                    break
                    
        if all_words_found:
            fallback_translation = " ".join(translated_words)
            return {"original": input_text, "translation": fallback_translation}

    # 3. AI Translation Fallback (Fine-Tuned Model)
    if model is None or tokenizer is None:
        if fallback_translation:
            return {"original": input_text, "translation": fallback_translation}
        raise HTTPException(status_code=503, detail="Translation model is not loaded.")

    lang_map = {"Yogad": "ilo_Latn", "Tagalog": "tgl_Latn", "English": "eng_Latn"}
    src_code = lang_map.get(request.src_lang, "ilo_Latn")
    tgt_code = lang_map.get(request.tgt_lang, "tgl_Latn")
    
    try:
        tokenizer.src_lang = src_code
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        forced_tgt_id = tokenizer.convert_tokens_to_ids(tgt_code)
        
        async with gpu_lock:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    forced_bos_token_id=forced_tgt_id, 
                    max_length=128,
                    num_beams=5,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
        
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"original": input_text, "translation": translation}
        
    except Exception as e:
        traceback.print_exc()
        if fallback_translation:
            return {"original": input_text, "translation": fallback_translation}
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        flush_vram()

@app.post("/translate_base")
async def translate_text_base(request: TranslationRequest):
    input_text = request.text.strip()
    if not input_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer is not loaded.")

    lang_map = {"Yogad": "eng_Latn", "Tagalog": "tgl_Latn", "English": "eng_Latn"}
    src_code = lang_map.get(request.src_lang, "eng_Latn")
    tgt_code = lang_map.get(request.tgt_lang, "tgl_Latn")
    
    try:
        b_model = get_base_nllb_model()
        tokenizer.src_lang = src_code
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        forced_tgt_id = tokenizer.convert_tokens_to_ids(tgt_code)
        
        async with gpu_lock:
            with torch.no_grad():
                outputs = b_model.generate(
                    **inputs, 
                    forced_bos_token_id=forced_tgt_id, 
                    max_length=128,
                    num_beams=5,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
        
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"original": input_text, "translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        flush_vram()

# --- 9. SUGGESTION ENDPOINT ---
@app.post("/submit_suggestion")
async def submit_suggestion(request: SuggestionRequest):
    src_text = request.src_text.strip()
    tgt_text = request.tgt_text.strip()
    if not src_text or not tgt_text:
        raise HTTPException(status_code=400, detail="Texts cannot be empty.")

    if request.src_lang == "Yogad" and request.tgt_lang == "Tagalog":
        yogad_word, tagalog_word = src_text, tgt_text
    elif request.src_lang == "Tagalog" and request.tgt_lang == "Yogad":
        yogad_word, tagalog_word = tgt_text, src_text
    else:
        raise HTTPException(status_code=400, detail="Invalid language pair.")

    pending_file = "Pending_Approvals.csv"
    try:
        if not os.path.exists(pending_file):
            with open(pending_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Action', 'Yogad', 'Tagalog', 'Status'])

        with open(pending_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([request.action_type, yogad_word, tagalog_word, "Pending Review"])
            
        return {"message": "Suggestion submitted for approval!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 10. WAV2VEC 2.0 ASR ENDPOINT ---
@app.post("/transcribe")
async def transcribe_speech(
    audio_file: UploadFile = File(...),
    src_lang: str = Form("Tagalog")
):
    try:
        audio_bytes = await audio_file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio recording received.")
            
        try:
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        except Exception:
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            waveform = torch.from_numpy(audio_array).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != 16000:
            waveform = F.resample(waveform, orig_freq=sr, new_freq=16000)
            
        duration_sec = waveform.shape[1] / 16000.0
        audio_array = waveform.squeeze(0).cpu().numpy()
        
        processor, asr_mod, err_msg = get_asr_engine(src_lang)
        if err_msg:
            return {
                "transcription": "",
                "language": src_lang,
                "duration": round(duration_sec, 2),
                "status": "pending_model",
                "message": err_msg
            }
            
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        
        async with gpu_lock:
            with torch.no_grad():
                logits = asr_mod(input_values).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        raw_transcription = processor.batch_decode(predicted_ids)[0]
        cleaned_transcription = re.sub(r'\s+', ' ', raw_transcription).strip()
        
        return {
            "transcription": cleaned_transcription,
            "language": src_lang,
            "duration": round(duration_sec, 2),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR Error: {str(e)}")
    finally:
        flush_vram()

# --- 11. HYBRID VITS TEXT-TO-SPEECH (TTS) ENDPOINT ---
@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    text_to_speak = request.text.strip()
    if not text_to_speak:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    helper, vits_model, err = get_vits_engine(request.lang)
    if err:
        return Response(
            content=f'{{"status": "pending_model", "message": "{err}"}}',
            media_type="application/json",
            status_code=503
        )

    try:
        wav_buffer = io.BytesIO()

        async with gpu_lock:
            with torch.no_grad():
                if request.lang == "Tagalog":
                    inputs = helper(text_to_speak, return_tensors="pt").to(device)
                    output = vits_model(**inputs)
                    waveform = output.waveform[0].cpu().float().numpy()
                    sf.write(wav_buffer, waveform, vits_model.config.sampling_rate, format="WAV")
                
                elif request.lang == "Yogad":
                    stn_tst = get_yogad_text(text_to_speak, helper)
                    x_tst = stn_tst.to(device).unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
                    
                    out = vits_model.infer(
                        x_tst, 
                        x_tst_lengths, 
                        noise_scale=0.667, 
                        noise_scale_w=0.8, 
                        length_scale=1.0
                    )
                    
                    audio_tensor = out[0]
                    audio = audio_tensor.data.cpu().float().numpy().flatten()
                    sf.write(wav_buffer, audio, helper.data.sampling_rate, format="WAV")

        wav_buffer.seek(0)
        return Response(content=wav_buffer.read(), media_type="audio/wav")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"VITS Synthesis Error: {str(e)}")
    finally:
        flush_vram()

@app.get("/")
def read_root():
    return {"status": "Yogad Bi-Directional AI Backend is Online"}
