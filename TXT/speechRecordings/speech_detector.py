import speech_recognition as sr
from os import path

r = sr.Recognizer()

with sr.Microphone() as source:
    print('Speak Anything')
    audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print(f'\nGoogle Speech Recognition thinks you said: \n\t\t{text}')

    except:
        print('Sorry could not recognize your voice.')

# print(path.dirname(path.realpath('speech_recognition.py')))
