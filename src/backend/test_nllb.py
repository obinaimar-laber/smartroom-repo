from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import ctranslate2

# Ganti ke model yang valid
MODEL_ID = "olob0/nllb-200-distilled-600M-ct2-int8_float16"

print("Downloading model...")
path = snapshot_download(MODEL_ID)
print(f"Model downloaded to: {path}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(path)

print("Loading model (INT8)...")
translator = ctranslate2.Translator(
    path,
    device="cuda",
    compute_type="int8_float16"
)

def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text).input_ids)
    result = translator.translate_batch(
        [tokens],
        target_prefix=[[tgt_lang]],
        max_batch_size=1
    )
    output_tokens = result[0].hypotheses[0][1:]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))

teks_rusia = "Привет, как дела? Меня зовут Абин."

print("\n=== Terjemahan ===")
print(f"Rusia asli : {teks_rusia}")
print(f"Indonesia  : {translate(teks_rusia, 'rus_Cyrl', 'ind_Latn')}")
print(f"Inggris    : {translate(teks_rusia, 'rus_Cyrl', 'eng_Latn')}")
print(f"Tionghoa   : {translate(teks_rusia, 'rus_Cyrl', 'zho_Hans')}")