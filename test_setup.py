"""
Test script to verify dependencies and system setup
Run this before starting the main application
"""

import sys

def test_imports():
    """Test all required imports"""
    print("🔍 Testing dependencies...\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test face_recognition
    try:
        import face_recognition
        print("✅ face_recognition:", face_recognition.__version__)
        tests_passed += 1
    except ImportError as e:
        print("❌ face_recognition:", str(e))
        tests_failed += 1
    
    # Test OpenCV
    try:
        import cv2
        print("✅ opencv-python:", cv2.__version__)
        tests_passed += 1
    except ImportError as e:
        print("❌ opencv-python:", str(e))
        tests_failed += 1
    
    # Test Gradio
    try:
        import gradio as gr
        print("✅ gradio:", gr.__version__)
        tests_passed += 1
    except ImportError as e:
        print("❌ gradio:", str(e))
        tests_failed += 1
    
    # Test NumPy
    try:
        import numpy as np
        print("✅ numpy:", np.__version__)
        tests_passed += 1
    except ImportError as e:
        print("❌ numpy:", str(e))
        tests_failed += 1
    
    # Test SciPy
    try:
        import scipy
        print("✅ scipy:", scipy.__version__)
        tests_passed += 1
    except ImportError as e:
        print("❌ scipy:", str(e))
        tests_failed += 1
    
    # Test scikit-learn
    try:
        import sklearn
        print("✅ scikit-learn:", sklearn.__version__)
        tests_passed += 1
    except ImportError as e:
        print("❌ scikit-learn:", str(e))
        tests_failed += 1
    
    # Test dlib (face_recognition dependency)
    try:
        import dlib
        print("✅ dlib:", dlib.__version__)
        tests_passed += 1
    except ImportError as e:
        print("❌ dlib:", str(e))
        tests_failed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests Passed: {tests_passed}/{tests_passed + tests_failed}")
    print(f"Tests Failed: {tests_failed}/{tests_passed + tests_failed}")
    
    if tests_failed > 0:
        print("\n⚠️  Some dependencies are missing!")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed correctly!")
        return True

def test_camera():
    """Test camera availability"""
    print("\n🔍 Testing camera access...\n")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not accessible (index 0)")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print(f"✅ Camera accessible (Resolution: {frame.shape[1]}x{frame.shape[0]})")
            return True
        else:
            print("❌ Camera opened but failed to read frame")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_face_detection():
    """Test basic face detection"""
    print("\n🔍 Testing face detection...\n")
    
    try:
        import face_recognition
        import numpy as np
        
        # Create a simple test image (blank)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # This should return empty list (no faces)
        locations = face_recognition.face_locations(test_image, model='hog')
        
        print(f"✅ Face detection working (found {len(locations)} faces in blank image)")
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_database():
    """Test SQLite database"""
    print("\n🔍 Testing database...\n")
    
    try:
        import sqlite3
        
        # Create temporary test database
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE test (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        ''')
        
        cursor.execute("INSERT INTO test (name) VALUES (?)", ("test",))
        conn.commit()
        
        cursor.execute("SELECT * FROM test")
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            print("✅ SQLite database working")
            return True
        else:
            print("❌ Database test failed")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Face Attendance System - Setup Verification")
    print("=" * 50 + "\n")
    
    print(f"Python Version: {sys.version}\n")
    
    results = []
    
    # Run tests
    results.append(("Dependencies", test_imports()))
    results.append(("Camera", test_camera()))
    results.append(("Face Detection", test_face_detection()))
    results.append(("Database", test_database()))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! System ready to run.")
        print("\nRun the application with:")
        print("  python main.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check camera connection and permissions")
        print("  - Ensure Python 3.8+ is installed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
