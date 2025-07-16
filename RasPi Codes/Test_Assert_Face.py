import unittest
import cv2
import os

class TestFaceDetection(unittest.TestCase):
    def setUp(self):
        # Use a known test image with a face in it
        self.test_image_path = r"D:\MECH\PYTHON\Engineering-teamwork---Rover\Project\TestImages\face1.jpg"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def test_face_detection_on_image(self):
        """Assert that at least one face is detected in the image"""
        self.assertTrue(os.path.exists(self.test_image_path), "Test image file not found")

        img = cv2.imread(self.test_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        self.assertGreater(len(faces), 0, "No face detected in the test image")

if __name__ == "__main__":
    unittest.main()


#Put a known image that contains a clear face into a folder like:
'''Confirms the test image exists.

Converts the image to grayscale.

Runs the Haar cascade detector.

Asserts that at least one face is found.'''