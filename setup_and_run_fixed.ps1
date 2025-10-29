<#
  setup_and_run_fixed.ps1
  Fixed version for Windows PowerShell (writes a small import_check.py file).
  Usage:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
    .\setup_and_run_fixed.ps1
#>

Write-Host "== setup_and_run_fixed.ps1 starting =="

# --- ensure py -3.11 exists ---
$py311_exists = $false
try {
    $ver = & py -3.11 --version 2>&1
    if ($LASTEXITCODE -eq 0) { $py311_exists = $true; Write-Host $ver }
    else { Write-Host "py -3.11 not found: $ver" -ForegroundColor Yellow }
} catch {
    Write-Host "py launcher unavailable or py -3.11 not installed." -ForegroundColor Red
}

if (-not $py311_exists) {
    Write-Host "`nERROR: Python 3.11 not found via 'py -3.11'. Install Python 3.11 and ensure it's registered with the py launcher." -ForegroundColor Red
    exit 1
}

# --- create venv if missing ---
$venvPath = ".\.venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "`n== Creating virtual environment (.venv) with Python 3.11 =="
    py -3.11 -m venv .venv
} else {
    Write-Host "`n== Using existing virtual environment .venv =="
}

# --- activate venv for this script ---
Write-Host "`n== Activating .venv =="
. .\.venv\Scripts\Activate.ps1

# --- confirm python version inside venv ---
Write-Host "`n== Python version in venv =="
python --version

# --- upgrade pip/setuptools/wheel ---
Write-Host "`n== Upgrading pip, setuptools, wheel =="
python -m pip install --upgrade pip setuptools wheel

# --- install pipwin to help with PyAudio on Windows ---
Write-Host "`n== Installing pipwin =="
python -m pip install pipwin

# --- attempt to install PyAudio via pipwin ---
Write-Host "`n== Attempting pipwin install pyaudio (if needed) =="
try {
    pipwin install pyaudio
} catch {
    Write-Host "pipwin pyaudio install failed (will try pip). Error: $_" -ForegroundColor Yellow
    try { python -m pip install pyaudio } catch { Write-Host "pip install pyaudio also failed (this is common). You can install a prebuilt wheel manually." -ForegroundColor Yellow }
}

# --- Install project requirements if requirements.txt exists ---
if (Test-Path ".\requirements.txt") {
    Write-Host "`n== Installing requirements.txt into venv =="
    python -m pip install -r .\requirements.txt
} else {
    Write-Host "`nWARNING: requirements.txt not found in current directory. Skipping pip install -r requirements.txt" -ForegroundColor Yellow
}

# --- Try to make ffmpeg available (winget/choco) ---
Write-Host "`n== Checking ffmpeg availability =="
$ff = & ffmpeg -version 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "ffmpeg already installed and in PATH."
} else {
    Write-Host "ffmpeg not found in PATH. Trying winget/choco..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing ffmpeg via winget (requires admin for system installs)..."
        winget install --id=Gyan.FFmpeg -e --source winget
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host "Installing ffmpeg via chocolatey..."
        choco install ffmpeg -y
    } else {
        Write-Host "winget/choco not available. Please install ffmpeg manually from https://www.gyan.dev/ffmpeg/builds/ and add ffmpeg\\bin to PATH." -ForegroundColor Yellow
    }
}

# --- Download TextBlob corpora if textblob is present ---
try {
    python -c "import importlib; importlib.import_module('textblob'); print('textblob present')"
    Write-Host "`n== Downloading textblob corpora =="
    python -m textblob.download_corpora >/dev/null 2>&1
    Write-Host "TextBlob corpora download attempted."
} catch {
    Write-Host "`nTextBlob not installed or available, skipping corpora download." -ForegroundColor Yellow
}

# --- write an import_check.py file and run it ---
Write-Host "`n== Writing import_check.py and running it =="
$importCheck = @"
import importlib, sys
pkgs = ['flask','flask_migrate','cv2','librosa','moviepy','speech_recognition','textblob','transformers','torch','deepface','noisereduce','soundfile']
print('Using', sys.executable)
for p in pkgs:
    spec = importlib.util.find_spec(p)
    print(('OK' if spec else 'MISSING').ljust(8), p)
"@
Set-Content -Path .\import_check.py -Value $importCheck -Encoding UTF8

Write-Host "`n== Running import_check.py =="
python .\import_check.py

# --- start the app ---
Write-Host "`n== Attempting to run python app.py (CTRL+C to stop) =="
try {
    python app.py
} catch {
    Write-Host "`nApp failed to start or exited. Check the output above for errors. If modules are MISSING (from the import check), install them inside this venv with: python -m pip install <package>" -ForegroundColor Yellow
}

Write-Host "`n== setup_and_run_fixed.ps1 finished =="
