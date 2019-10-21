import pyttsx3
engine = pyttsx3.init()

# definindo voz como masculina
# sound = engine.getProperty('voices')

# engine.setProperty('voice', sound[0].id)
engine.say('hello meu consagrado')

engine.runAndWait()
# conclusao: pyttsx3 melhor que google tts