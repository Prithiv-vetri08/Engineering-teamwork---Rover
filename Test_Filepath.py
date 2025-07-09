import unittest
import os
from dataset_loader import load_dataset

class TestDatasetPath(unittest.TestCase):
    def setUp(self):
        # Change this to match your actual dataset path
        self.valid_path = r"D:\MECH\PYTHON\Engineering-teamwork---Rover\Project"
        self.invalid_path = r"D:\Invalid\Path\To\Dataset"

    def test_valid_dataset_path_exists(self):
        """Check if the dataset path exists and is a directory"""
        self.assertTrue(os.path.exists(self.valid_path), "Dataset path does not exist")
        self.assertTrue(os.path.isdir(self.valid_path), "Dataset path is not a directory")

    def test_valid_dataset_contains_subfolders(self):
        """Check that the dataset directory has at least one subfolder"""
        subdirs = [d for d in os.listdir(self.valid_path) if os.path.isdir(os.path.join(self.valid_path, d))]
        self.assertGreater(len(subdirs), 0, "No subfolders (people) found in dataset")

    def test_invalid_dataset_path_raises_error(self):
        """Check that loading from an invalid path raises an error"""
        with self.assertRaises(FileNotFoundError):
            load_dataset(self.invalid_path)

if __name__ == "__main__":
    unittest.main()
    