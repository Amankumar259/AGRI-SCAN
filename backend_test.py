import requests
import sys
import json
import io
from datetime import datetime
from PIL import Image

class PlantDiseaseAPITester:
    def __init__(self, base_url="https://plant-defender-9.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details
        })

    def create_test_image(self):
        """Create a simple test image"""
        img = Image.new('RGB', (224, 224), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes

    def test_health_check(self):
        """Test health check endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f", Service: {data.get('service', 'Unknown')}"
            self.log_test("Health Check", success, details)
            return success
        except Exception as e:
            self.log_test("Health Check", False, str(e))
            return False

    def test_root_endpoint(self):
        """Test root API endpoint"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f", Message: {data.get('message', 'No message')}"
            self.log_test("Root Endpoint", success, details)
            return success
        except Exception as e:
            self.log_test("Root Endpoint", False, str(e))
            return False

    def test_predict_endpoint_valid_image(self):
        """Test prediction endpoint with valid image"""
        try:
            test_image = self.create_test_image()
            files = {'file': ('test_plant.jpg', test_image, 'image/jpeg')}
            
            response = requests.post(f"{self.api_url}/predictions/predict", files=files, timeout=30)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                required_fields = ['prediction_id', 'filename', 'predicted_disease', 'confidence', 'all_predictions', 'timestamp']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    success = False
                    details += f", Missing fields: {missing_fields}"
                else:
                    details += f", Disease: {data['predicted_disease']}, Confidence: {data['confidence']:.2f}"
                    # Store prediction_id for history test
                    self.last_prediction_id = data.get('prediction_id')
            else:
                try:
                    error_data = response.json()
                    details += f", Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    details += f", Response: {response.text[:100]}"
            
            self.log_test("Predict Valid Image", success, details)
            return success
        except Exception as e:
            self.log_test("Predict Valid Image", False, str(e))
            return False

    def test_predict_endpoint_invalid_file(self):
        """Test prediction endpoint with invalid file type"""
        try:
            # Create a text file instead of image
            files = {'file': ('test.txt', io.StringIO('This is not an image'), 'text/plain')}
            
            response = requests.post(f"{self.api_url}/predictions/predict", files=files, timeout=10)
            success = response.status_code == 415  # Unsupported Media Type expected
            details = f"Status: {response.status_code}"
            
            if response.status_code == 415:
                try:
                    error_data = response.json()
                    details += f", Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    pass
            
            self.log_test("Predict Invalid File Type", success, details)
            return success
        except Exception as e:
            self.log_test("Predict Invalid File Type", False, str(e))
            return False

    def test_predict_endpoint_no_file(self):
        """Test prediction endpoint without file"""
        try:
            response = requests.post(f"{self.api_url}/predictions/predict", timeout=10)
            success = response.status_code == 422  # Unprocessable Entity expected
            details = f"Status: {response.status_code}"
            
            self.log_test("Predict No File", success, details)
            return success
        except Exception as e:
            self.log_test("Predict No File", False, str(e))
            return False

    def test_history_endpoint(self):
        """Test history endpoint"""
        try:
            response = requests.get(f"{self.api_url}/predictions/history", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                if 'total' in data and 'results' in data:
                    details += f", Total: {data['total']}, Results: {len(data['results'])}"
                else:
                    success = False
                    details += ", Missing required fields (total, results)"
            
            self.log_test("History Endpoint", success, details)
            return success
        except Exception as e:
            self.log_test("History Endpoint", False, str(e))
            return False

    def test_history_with_pagination(self):
        """Test history endpoint with pagination"""
        try:
            response = requests.get(f"{self.api_url}/predictions/history?limit=5&skip=0", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                results = data.get('results', [])
                details += f", Results count: {len(results)}"
                
                # Check if results have required fields
                if results:
                    first_result = results[0]
                    required_fields = ['filename', 'predicted_disease', 'confidence', 'timestamp']
                    missing_fields = [field for field in required_fields if field not in first_result]
                    if missing_fields:
                        success = False
                        details += f", Missing fields in results: {missing_fields}"
            
            self.log_test("History Pagination", success, details)
            return success
        except Exception as e:
            self.log_test("History Pagination", False, str(e))
            return False

    def run_all_tests(self):
        """Run all backend tests"""
        print("🧪 Starting Plant Disease API Tests...")
        print(f"🌐 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Test basic connectivity
        if not self.test_health_check():
            print("❌ Health check failed - API may be down")
            return False
        
        self.test_root_endpoint()
        
        # Test prediction endpoints
        self.test_predict_endpoint_valid_image()
        self.test_predict_endpoint_invalid_file()
        self.test_predict_endpoint_no_file()
        
        # Test history endpoints
        self.test_history_endpoint()
        self.test_history_with_pagination()
        
        # Print summary
        print("=" * 60)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed - check logs above")
            return False

def main():
    tester = PlantDiseaseAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())