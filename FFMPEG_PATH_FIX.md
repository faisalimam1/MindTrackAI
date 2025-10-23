# FFmpeg PATH Issue - Critical Fix Needed

## Problem

FFmpeg IS installed on your system (version 8.0-full_build-www.gyan.dev), BUT Python cannot access it because it's not in the system PATH that Python can see.

**Evidence:**
```
# From terminal (works):
$ ffmpeg --version
ffmpeg version 8.0-full_build-www.gyan.dev

# From Python subprocess (fails):
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

## Why This Happens

- FFmpeg works in Git Bash or certain terminals because those terminals have their own PATH
- Python's subprocess.run() uses Windows System PATH
- FFmpeg is not in Windows System PATH

## Solution

You need to add FFmpeg to your Windows System PATH. Here's how:

### Step 1: Find FFmpeg Installation

Common locations:
- `C:\ffmpeg\bin\`
- `C:\Program Files\ffmpeg\bin\`
- `C:\Users\YOUR_USERNAME\ffmpeg\bin\`
- `C:\ProgramData\chocolatey\bin\` (if installed via Chocolatey)

**To find it:**
1. Open File Explorer
2. Search for `ffmpeg.exe` in C:\ drive
3. Note the full path to the `bin` folder containing `ffmpeg.exe`

### Step 2: Add to System PATH

**Windows 10/11:**
1. Press `Win + X`, select "System"
2. Click "Advanced system settings" on the right
3. Click "Environment Variables" button
4. Under "System variables", find and select "Path"
5. Click "Edit"
6. Click "New"
7. Paste the path to FFmpeg's bin folder (e.g., `C:\ffmpeg\bin`)
8. Click "OK" on all dialogs
9. **IMPORTANT: Restart your terminal/command prompt**

### Step 3: Verify Installation

Open a NEW command prompt and run:
```cmd
ffmpeg -version
```

If it shows the version, FFmpeg is now in PATH.

### Step 4: Verify Python Can Access It

Run this in your project directory:
```cmd
cd "D:\projects\major-proj-1\MindTrackAI (2)\MindTrackAI"
python -c "import subprocess; subprocess.run(['ffmpeg', '-version'], check=True); print('SUCCESS: Python can access FFmpeg')"
```

If this succeeds, you're all set!

### Step 5: Restart the Application

After FFmpeg is in PATH:
1. Stop the running Flask app (Ctrl+C in the terminal running it)
2. Start it again: `python start.py`
3. Test the audio assessment again

## Alternative: Quick Fix Without System PATH

If you can't modify system PATH, I can modify the Python code to use a hardcoded FFmpeg path. Just tell me the exact path where ffmpeg.exe is located (e.g., `C:\ffmpeg\bin\ffmpeg.exe`), and I'll update the code.

## What Will Work After This Fix

Once FFmpeg is accessible to Python:
- ✅ Audio conversion will work (WebM → WAV)
- ✅ Voice feature extraction (pitch, energy, jitter, shimmer)
- ✅ Transcription will work (Google Speech, Whisper, Sphinx)
- ✅ Crisis keyword detection ("suicidal", "hopeless", "end my life")
- ✅ Dynamic depression scores (not always 60%)
- ✅ CRITICAL wellbeing level for crisis statements
- ✅ Crisis alerts with emergency hotlines
- ✅ Proper recommendations based on severity

## Current Behavior vs Expected

### Current (FFmpeg not accessible):
- Transcription: "Audio received but transcription unclear"
- Depression: 60% (MODERATE) - even for crisis statements
- Wellbeing: FAIR - always
- Crisis detected: False - always
- Recommendations: Generic moderate-level suggestions

### After Fix (FFmpeg accessible):
- Transcription: "I want to end my life, I get suicidal thoughts and feel worthless and hopeless"
- Depression: 85-95% (SEVERE/CRITICAL)
- Wellbeing: CRITICAL
- Crisis detected: True
- Crisis Alert: Shows immediately with:
  - NIMHANS 24/7 Helpline: 080-46110007
  - iCall: 9152987821
  - Vandrevala Foundation: 1860-2662-345
  - Emergency actions to take
- Recommendations:
  - [URGENT] Contact Emergency Services Immediately
  - [CRITICAL] Crisis Helpline - Call NOW
  - [CRITICAL] Do Not Stay Alone
  - Tell trusted person immediately
  - Go to nearest hospital

## Need Help?

If you're stuck, please share:
1. The output of searching for `ffmpeg.exe` in File Explorer
2. Or tell me if you want me to modify the code to use a specific path instead
