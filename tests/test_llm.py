"""
Test basic functionality of the LLM module without requiring heavy dependencies.

This test validates the data preprocessing and basic structure without training.
"""

import unittest
import tempfile
import os
from loda.llm.data_preprocessing import DataPreprocessor, TrainingExample


class TestLodaLLM(unittest.TestCase):
    """Test basic LLM functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory with sample LODA programs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample LODA program files
        self.create_sample_program("A000045", 
            "; A000045: Fibonacci numbers: F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1.\n"
            "; Submitted by loader3229\n"
            "; 0,1,1,2,3,5,8,13,21,34,55,89\n"
            "mov $1,$0\n"
            "lpb $0\n"
            "  add $1,$2\n"
            "  mov $2,$1\n"
            "  sub $0,1\n"
            "lpe\n"
            "mov $0,$2"
        )
        
        self.create_sample_program("A000290",
            "; A000290: The squares: a(n) = n^2.\n"
            "; 0,1,4,9,16,25,36,49,64,81,100\n"
            "pow $0,2"
        )
    
    def create_sample_program(self, program_id, content):
        """Create a sample program file."""
        # Create subdirectory structure like programs/oeis/000/
        subdir = os.path.join(self.temp_dir, program_id[:3])
        os.makedirs(subdir, exist_ok=True)
        
        file_path = os.path.join(subdir, f"{program_id}.asm")
        with open(file_path, 'w') as f:
            f.write(content)
    
    def test_data_preprocessor_initialization(self):
        """Test DataPreprocessor can be initialized."""
        preprocessor = DataPreprocessor(self.temp_dir)
        self.assertIsNotNone(preprocessor)
        self.assertEqual(preprocessor.programs_dir, self.temp_dir)
    
    def test_extract_description_from_program(self):
        """Test description extraction from program text."""
        preprocessor = DataPreprocessor(self.temp_dir)
        
        program_text = (
            "; A000045: Fibonacci numbers: F(n) = F(n-1) + F(n-2) with F(0) = 0 and F(1) = 1.\n"
            "mov $1,$0\n"
        )
        
        description = preprocessor.extract_description_from_program(program_text)
        self.assertIsNotNone(description)
        self.assertIn("Fibonacci", description)
    
    def test_extract_terms_from_program(self):
        """Test sequence terms extraction."""
        preprocessor = DataPreprocessor(self.temp_dir)
        
        program_text = (
            "; A000290: The squares\n"
            "; 0,1,4,9,16,25,36,49\n"
            "pow $0,2\n"
        )
        
        terms = preprocessor.extract_terms_from_program(program_text)
        self.assertIsNotNone(terms)
        self.assertEqual(terms[:4], [0, 1, 4, 9])
    
    def test_clean_loda_code(self):
        """Test LODA code cleaning."""
        preprocessor = DataPreprocessor(self.temp_dir)
        
        dirty_code = (
            "; This is a comment\n"
            "mov $1,$0\n"
            "; Another comment\n"
            "pow $1,2\n"
            "\n"
            "mov $0,$1\n"
        )
        
        clean_code = preprocessor.clean_loda_code(dirty_code)
        expected = "mov $1,$0\npow $1,2\nmov $0,$1"
        self.assertEqual(clean_code, expected)
        
        # Test inline comment removal
        code_with_inline_comments = (
            "; Header comment\n"
            "add $0,1\n"
            "sub $0,2 ; inline comment here\n"
            "mul $1,3   ; another inline comment\n"
            "; Full line comment\n"
            "div $2,4\n"
        )
        
        clean_inline = preprocessor.clean_loda_code(code_with_inline_comments)
        expected_inline = "add $0,1\nsub $0,2\nmul $1,3\ndiv $2,4"
        self.assertEqual(clean_inline, expected_inline)
    
    def test_training_example_creation(self):
        """Test TrainingExample creation."""
        example = TrainingExample(
            sequence_id="A000001",
            description="Test sequence",
            loda_code="mov $0,1",
            terms=[1, 1, 1, 1]
        )
        
        self.assertEqual(example.sequence_id, "A000001")
        self.assertEqual(example.description, "Test sequence")
        self.assertEqual(example.loda_code, "mov $0,1")
        self.assertEqual(example.terms, [1, 1, 1, 1])
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()