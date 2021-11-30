from gtts import gTTS
from textblob import TextBlob
from translate import Translator
import os

def load_text(file):
    with open(file,encoding='utf-8') as f:
        text = f.read()
    return text.lower()


text = load_text("summary.txt")

def translate_lang(sumer):
    t = TextBlob(sumer)
    t = t.detect_language()
    lang = input()
    translator= Translator(from_lang=t,to_lang=lang)
    translation = translator.translate(sumer)
    print(translation)
    if lang == 'english':
        language = 'en'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("reportaudio.mp3")

    elif lang == 'french':
        language = 'fr'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("reportaudio.mp3")

    elif lang == 'german':
        language = 'de'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("reportaudio.mp3")

    elif lang == 'chinese':
        language = 'zh-TW'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("reportaudio.mp3")

    elif lang == 'arabic':
        language = 'ar'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("reportaudio.mp3")

    elif lang == 'spanish':
        language = 'es'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("reportaudio.mp3")

    elif lang == 'russian':
        language = 'ru'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("reportaudio.mp3")

    elif lang == 'hindi':
        language = 'hi'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("reportaudio.mp3")

    elif lang == 'japanese':
        language = 'ja'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("reportaudio.mp3")

    elif lang == 'korean':
        language = 'ko'
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.save("reportaudio.mp3")

    else:
        print("Enter a Valid Language")

translate_lang(text)