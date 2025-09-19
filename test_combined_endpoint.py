#!/usr/bin/env python3
"""
Test script to verify the combined assessment endpoint is working
"""

import requests
import json

def test_combined_assessment_endpoint():
    """Test the combined assessment endpoint"""
    base_url = "http://localhost:5000"  # Adjust if your server runs on different port
    
    print("=== TESTING COMBINED ASSESSMENT ENDPOINT ===")
    
    # Test 1: Test endpoint (no authentication required)
    print("\n1. Testing combined assessment endpoint...")
    try:
        response = requests.post(f"{base_url}/assessment/api/test-combined-assessment")
        if response.status_code == 200:
            data = response.json()
            print("✅ Test endpoint working!")
            print(f"   Success: {data.get('success')}")
            print(f"   Message: {data.get('message')}")
            print(f"   PHQ-9 Available: {data.get('phq9_available')}")
            print(f"   SCID-5 Available: {data.get('scid5_available')}")
            
            composite = data.get('composite_assessment', {})
            print(f"   Depression Score: {composite.get('depression_score', 'N/A')}")
            print(f"   Depression Level: {composite.get('depression_level', 'N/A')}")
            print(f"   Confidence Score: {composite.get('confidence_score', 'N/A')}")
            print(f"   Overall Risk: {composite.get('overall_risk', 'N/A')}")
            print(f"   Completeness: {composite.get('completeness_score', 'N/A')}")
        else:
            print(f"❌ Test endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
    
    # Test 2: Test with mock audio file
    print("\n2. Testing with mock audio file...")
    try:
        # Create a simple test audio file (just some bytes)
        test_audio_data = b"test audio data for analysis"
        
        files = {'recording': ('test_audio.wav', test_audio_data, 'audio/wav')}
        data = {'assessment_type': 'audio-only'}
        
        response = requests.post(f"{base_url}/assessment/api/analyze-recording", files=files, data=data)
        if response.status_code == 200:
            result = response.json()
            print("✅ Audio analysis endpoint working!")
            print(f"   Depression Score: {result.get('depression_score', 'N/A')}")
            print(f"   Depression Level: {result.get('depression_level', 'N/A')}")
            print(f"   Confidence Score: {result.get('confidence_score', 'N/A')}")
            print(f"   Overall Risk: {result.get('overall_risk', 'N/A')}")
            print(f"   Completeness: {result.get('completeness_score', 'N/A')}")
            
            # Check if combined assessment is included
            if 'composite_assessment' in result:
                print("✅ Combined assessment included in results!")
                composite = result['composite_assessment']
                print(f"   Combined Depression: {composite.get('depression_score', 'N/A')}")
                print(f"   Combined Risk: {composite.get('overall_risk', 'N/A')}")
            else:
                print("⚠️ Combined assessment not found in results")
        else:
            print(f"❌ Audio analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing audio analysis: {e}")
    
    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    test_combined_assessment_endpoint()
