import pytts3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    # engine.setProperty('voice', 'com.apple.speech.synthesis.voice.sin-ji') # 粤语
    engine.setProperty('voice', voice.id)  # 循环设置各种语音播报的人声
    engine.say('一行数据')
engine.runAndWait()