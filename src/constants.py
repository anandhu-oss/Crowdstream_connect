"""Shared constants. Qwen3-ASR language list checked against the model card (30 languages)."""
SR = 16000

# Names Qwen3-ASR uses -> ISO codes (Hugging Face model card, Qwen/Qwen3-ASR-1.7B).
QWEN_LANGUAGES = {
    "Chinese": "zh", "English": "en", "Cantonese": "yue", "Arabic": "ar", "German": "de", "French": "fr",
    "Spanish": "es", "Portuguese": "pt", "Indonesian": "id", "Italian": "it", "Korean": "ko", "Russian": "ru",
    "Thai": "th", "Vietnamese": "vi", "Japanese": "ja", "Turkish": "tr", "Hindi": "hi", "Malay": "ms",
    "Dutch": "nl", "Swedish": "sv", "Danish": "da", "Finnish": "fi", "Polish": "pl", "Czech": "cs",
    "Filipino": "fil", "Persian": "fa", "Greek": "el", "Hungarian": "hu", "Macedonian": "mk", "Romanian": "ro",
}
CODE_TO_QWEN = {code: name for name, code in QWEN_LANGUAGES.items()}

# A label that real recordings showed is sometimes wrong: Qwen called Malayalam speech "Malay" in our tests.
# The code is kept, but the result is marked uncertain with these as the possible true languages.
CONFUSABLE = {"ms": ("ml",)}
