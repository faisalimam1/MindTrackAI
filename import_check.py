import importlib, sys
pkgs = ['flask','flask_migrate','cv2','librosa','moviepy','speech_recognition','textblob','transformers','torch','deepface','noisereduce','soundfile']
print('Using', sys.executable)
for p in pkgs:
    spec = importlib.util.find_spec(p)
    print(('OK' if spec else 'MISSING').ljust(8), p)
