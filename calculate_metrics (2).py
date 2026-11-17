import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from peft import PeftModel
import jiwer
import sacrebleu
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#python -m streamlit run app.py
#python -m uvicorn main:app --reload

print("--- YOGAD AI METRICS ENGINE ---")

# Load the Model and Tokenizer
print("Loading V7 Tourist Model...")
base_model_id = "facebook/nllb-200-distilled-1.3B"

# UPDATED: Pointing to the new NLLB weights folder
adapter_path = "./Yogad_NLLB_Final"

#fallback language
tokenizer = NllbTokenizer.from_pretrained(adapter_path, src_lang="ilo_Latn", tgt_lang="tgl_Latn")
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, adapter_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# 2. Load the Test Data
print("Loading CSV Data...")
df = pd.read_csv("Yogad_Tourism_Corpus_upd - Sheet1.csv")   

# We take a random sample of 100 rows for the evaluation
# (You can increase n=100 to evaluate on more data if you want)
test_df = df.sample(n=100, random_state=42).reset_index(drop=True)

references = []  # The actual human Tagalog translation
hypotheses = []  # The AI's guessed Tagalog translation

# 3. Give the AI the Test Exam
print("Running AI Translations (This may take a few minutes)...")
for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    yogad_text = str(row['Yogad'])
    true_tagalog = str(row['Tagalog']).strip()
    
    # Generate translation
    inputs = tokenizer(yogad_text, return_tensors="pt").to(device)
    forced_tgt_id = tokenizer.convert_tokens_to_ids("tgl_Latn")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            forced_bos_token_id=forced_tgt_id, 
            max_length=128,
            num_beams=5
        )
    
    predicted_tagalog = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    references.append(true_tagalog)
    hypotheses.append(predicted_tagalog)

# 4. Calculate Official Metrics
print("\n--- OFFICIAL THESIS METRICS ---")

# Calculate Word Error Rate (WER)
wer = jiwer.wer(references, hypotheses)

# Calculate Character Error Rate (CER)
cer = jiwer.cer(references, hypotheses)

# Calculate BLEU Score
# sacrebleu expects a list of reference lists
bleu = sacrebleu.corpus_bleu(hypotheses, [references])

# Calculate CHRF Score
chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

print(f"✅ WER (Word Error Rate):       {wer * 100:.2f}%")
print(f"✅ CER (Character Error Rate):  {cer * 100:.2f}%")
print(f"✅ BLEU Score:                  {bleu.score:.2f}")
print(f"✅ CHRF Score:                  {chrf.score:.2f}")

print("\n(Note for Defense: Lower is better for WER/CER. Higher is better for BLEU.)")