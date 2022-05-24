import pyttsx3

engine = pyttsx3.init(debug=True)
# volume = engine.getProperty('volume')
engine.setProperty('volume', 1.0)
# engine.say(self.txt)
engine.save_to_file("您好！", "Hello.mp3")
engine.runAndWait()
engine.stop()