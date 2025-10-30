import requests
import json
import base64
import os
from PIL import Image
import io

# Base URL for the FastAPI backend
BASE_URL = "http://localhost:8000"

def test_student_endpoints():
    print("\n=== Testing Student Endpoints ===")
    
    # Test get all students
    response = requests.get(f"{BASE_URL}/students")
    print(f"Get all students status: {response.status_code}")
    if response.status_code == 200:
        students = response.json()
        print(f"Found {len(students)} students")
    
    # Test attendance stats
    response = requests.get(f"{BASE_URL}/attendance-stats")
    print(f"Get attendance stats status: {response.status_code}")
    if response.status_code == 200:
        stats = response.json()
        print(f"Total attendance records: {stats.get('total_records', 'N/A')}")

def test_qr_code():
    print("\n=== Testing QR Code Generation ===")
    
    # Test QR code generation
    test_data = {"student_name": "Test Student"}
    response = requests.post(f"{BASE_URL}/qr-code", json=test_data)
    print(f"QR code generation status: {response.status_code}")
    
    if response.status_code == 200:
        qr_data = response.json()
        print(f"QR code generated: {'qr_code_base64' in qr_data}")

def test_report_generation():
    print("\n=== Testing Report Generation ===")
    
    # Test report generation
    test_data = {
        "report_type": "daily",
        "start_date": "2023-01-01",
        "end_date": "2023-01-01"
    }
    
    response = requests.post(f"{BASE_URL}/generate-report", json=test_data)
    print(f"Report generation status: {response.status_code}")
    
    # For PDF responses, check content type
    if response.status_code == 200:
        content_type = response.headers.get('content-type', '')
        print(f"Report content type: {content_type}")
        print(f"Report size: {len(response.content)} bytes")

def test_ai_insights():
    print("\n=== Testing AI Insights ===")
    
    # Test AI insights generation
    response = requests.get(f"{BASE_URL}/ai-insights")
    print(f"AI insights generation status: {response.status_code}")
    
    if response.status_code == 200:
        insights = response.json()
        print(f"AI insights generated: {'insights' in insights}")
        if 'insights' in insights:
            print(f"Summary available: {'summary' in insights['insights']}")

def run_all_tests():
    print("Starting API tests...")
    
    try:
        test_student_endpoints()
        test_qr_code()
        test_report_generation()
        test_ai_insights()
        
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    run_all_tests()