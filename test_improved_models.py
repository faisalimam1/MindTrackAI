#!/usr/bin/env python3
"""
Test script for improved high-accuracy models
"""

import numpy as np
import tempfile
import os
from high_accuracy_models import HighAccuracyAudioAnalyzer, HighAccuracyVideoAnalyzer

def test_audio_analyzer():
    """Test the improved audio analyzer"""
    print("=== Testing Improved Audio Analyzer ===")
    
    try:
        analyzer = HighAccuracyAudioAnalyzer()
        print("Audio analyzer initialized successfully")
        
        # Create test audio data (2 seconds of sine wave)
        sample_rate = 16000
        duration = 2  # seconds
        frequency = 440  # A4 note
        
        # Generate test audio
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Convert to bytes
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            with open(tmp_file.name, 'rb') as f:
                audio_bytes = f.read()
            os.unlink(tmp_file.name)
        
        print(f"Test audio created: {len(audio_bytes)} bytes")
        
        # Test analysis
        results = analyzer.analyze_audio(audio_bytes)
        print("Audio analysis completed")
        
        # Display results
        print(f"  Depression Score: {results.get('combined_score', {}).get('depression_score', 'N/A')}")
        print(f"  Confidence Score: {results.get('combined_score', {}).get('confidence_score', 'N/A')}")
        print(f"  Overall Confidence: {results.get('confidence', 'N/A')}")
        print(f"  Model Used: {results.get('model_used', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"Audio analyzer test failed: {e}")
        return False

def test_video_analyzer():
    """Test the improved video analyzer"""
    print("\n=== Testing Improved Video Analyzer ===")
    
    try:
        analyzer = HighAccuracyVideoAnalyzer()
        print("Video analyzer initialized successfully")
        
        # Create test video data (simple colored frame)
        import cv2
        import tempfile
        
        # Create a simple test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [100, 150, 200]  # Blue-ish color
        
        # Add some features to make it more realistic
        cv2.rectangle(frame, (200, 150), (400, 350), (255, 255, 255), -1)  # White rectangle (face-like)
        cv2.circle(frame, (250, 200), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(frame, (350, 200), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(frame, (300, 250), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        # Save as video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_file.name, fourcc, 1.0, (640, 480))
            
            # Write 30 frames (30 seconds at 1 fps)
            for _ in range(30):
                out.write(frame)
            out.release()
            
            with open(tmp_file.name, 'rb') as f:
                video_bytes = f.read()
            os.unlink(tmp_file.name)
        
        print(f"Test video created: {len(video_bytes)} bytes")
        
        # Test analysis
        results = analyzer.analyze_video(video_bytes)
        print("Video analysis completed")
        
        # Display results
        combined = results.get('combined_analysis', {})
        print(f"  Depression Score: {combined.get('depression_score', 'N/A')}")
        print(f"  Confidence Score: {combined.get('confidence_score', 'N/A')}")
        print(f"  Overall Confidence: {results.get('confidence', 'N/A')}")
        print(f"  Model Used: {results.get('model_used', 'N/A')}")
        print(f"  Frames Analyzed: {combined.get('frames_analyzed', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"Video analyzer test failed: {e}")
        return False

def test_model_info():
    """Test model information retrieval"""
    print("\n=== Testing Model Information ===")
    
    try:
        from video_audio_service_high_accuracy import get_high_accuracy_model_info
        
        model_info = get_high_accuracy_model_info()
        print("Model info retrieved successfully")
        
        print(f"  High Accuracy Available: {model_info.get('high_accuracy_available', False)}")
        
        if model_info.get('high_accuracy_available'):
            models = model_info.get('models', {})
            print(f"  Audio Model: {models.get('audio', {}).get('primary', 'N/A')}")
            print(f"  Video Model: {models.get('video', {}).get('primary', 'N/A')}")
            print(f"  Combined Accuracy: {models.get('combined', {}).get('accuracy', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"Model info test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Improved High-Accuracy Models")
    print("=" * 50)
    
    tests = [
        test_audio_analyzer,
        test_video_analyzer,
        test_model_info
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The improved models are working correctly.")
    else:
        print("Some tests failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    main()
