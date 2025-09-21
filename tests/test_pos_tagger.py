"""
Comprehensive test suite for Part-of-Speech tagging project
Tests all components including tagger, web app, and utilities
"""

import pytest
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pos_tagger import AdvancedPOSTagger, POSTag
from data.mock_database import MockDatabase, SampleText
from utils.visualizer import POSVisualizer

class TestAdvancedPOSTagger:
    """Test cases for AdvancedPOSTagger class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tagger = AdvancedPOSTagger(language="en")
        self.sample_text = "The quick brown fox jumps over the lazy dog."
    
    def test_tagger_initialization(self):
        """Test tagger initialization"""
        assert self.tagger.language == "en"
        assert self.tagger.nlp is not None
        assert len(self.tagger.supported_languages) > 0
    
    def test_tag_text_basic(self):
        """Test basic text tagging"""
        tags = self.tagger.tag_text(self.sample_text)
        
        assert len(tags) > 0
        assert all(isinstance(tag, POSTag) for tag in tags)
        
        # Check that we have expected POS tags
        pos_tags = [tag.pos for tag in tags if not tag.is_space and not tag.is_punct]
        assert "DET" in pos_tags  # "The"
        assert "ADJ" in pos_tags  # "quick", "brown", "lazy"
        assert "NOUN" in pos_tags  # "fox", "dog"
        assert "VERB" in pos_tags  # "jumps"
    
    def test_tag_text_with_confidence(self):
        """Test text tagging with confidence scores"""
        tags = self.tagger.tag_text(self.sample_text, include_confidence=True)
        
        assert all(tag.confidence is not None for tag in tags)
        assert all(0 <= tag.confidence <= 1 for tag in tags if tag.confidence is not None)
    
    def test_get_pos_statistics(self):
        """Test POS statistics calculation"""
        tags = self.tagger.tag_text(self.sample_text)
        stats = self.tagger.get_pos_statistics(tags)
        
        assert "total_words" in stats
        assert "pos_counts" in stats
        assert "pos_percentages" in stats
        assert "unique_pos_tags" in stats
        
        assert stats["total_words"] > 0
        assert stats["unique_pos_tags"] > 0
        assert sum(stats["pos_percentages"].values()) == pytest.approx(100.0, abs=0.1)
    
    def test_extract_phrases(self):
        """Test phrase extraction"""
        phrases = self.tagger.extract_phrases(self.sample_text)
        
        assert isinstance(phrases, list)
        # Should have at least one noun phrase
        noun_phrases = [p for p in phrases if p["type"] == "noun_phrase"]
        assert len(noun_phrases) > 0
    
    def test_analyze_sentence_structure(self):
        """Test sentence structure analysis"""
        structure = self.tagger.analyze_sentence_structure(self.sample_text)
        
        assert "root" in structure
        assert "root_pos" in structure
        assert "dependencies" in structure
        assert "sentence_length" in structure
        
        assert structure["root"] is not None
        assert structure["sentence_length"] > 0
        assert len(structure["dependencies"]) > 0
    
    def test_multilingual_support(self):
        """Test multilingual language support"""
        # Test Spanish
        spanish_tagger = AdvancedPOSTagger(language="es")
        assert spanish_tagger.language == "es"
        
        # Test French
        french_tagger = AdvancedPOSTagger(language="fr")
        assert french_tagger.language == "fr"
    
    def test_empty_text(self):
        """Test handling of empty text"""
        tags = self.tagger.tag_text("")
        assert len(tags) == 0
    
    def test_single_word(self):
        """Test tagging of single word"""
        tags = self.tagger.tag_text("hello")
        assert len(tags) == 1
        assert tags[0].word == "hello"
        assert tags[0].pos in ["INTJ", "NOUN", "VERB"]  # Could be any of these

class TestMockDatabase:
    """Test cases for MockDatabase class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.db = MockDatabase()
    
    def test_database_initialization(self):
        """Test database initialization"""
        assert len(self.db.samples) > 0
        assert all(isinstance(sample, SampleText) for sample in self.db.samples)
    
    def test_get_all_samples(self):
        """Test getting all samples"""
        samples = self.db.get_all_samples()
        assert len(samples) > 0
        assert all("id" in sample for sample in samples)
        assert all("text" in sample for sample in samples)
    
    def test_get_samples_by_domain(self):
        """Test filtering samples by domain"""
        news_samples = self.db.get_samples_by_domain("news")
        assert len(news_samples) > 0
        assert all(sample["domain"] == "news" for sample in news_samples)
    
    def test_get_samples_by_language(self):
        """Test filtering samples by language"""
        en_samples = self.db.get_samples_by_language("en")
        assert len(en_samples) > 0
        assert all(sample["language"] == "en" for sample in en_samples)
    
    def test_get_sample_by_id(self):
        """Test getting sample by ID"""
        sample = self.db.get_sample_by_id("news_001")
        assert sample is not None
        assert sample["id"] == "news_001"
        
        # Test non-existent ID
        sample = self.db.get_sample_by_id("nonexistent")
        assert sample is None
    
    def test_search_samples(self):
        """Test searching samples"""
        ai_samples = self.db.search_samples("AI")
        assert len(ai_samples) > 0
        
        # Test case insensitive search
        ai_samples_lower = self.db.search_samples("ai")
        assert len(ai_samples_lower) == len(ai_samples)
    
    def test_get_domains(self):
        """Test getting available domains"""
        domains = self.db.get_domains()
        assert len(domains) > 0
        assert "news" in domains
        assert "literature" in domains
    
    def test_get_languages(self):
        """Test getting available languages"""
        languages = self.db.get_languages()
        assert len(languages) > 0
        assert "en" in languages
    
    def test_get_difficulties(self):
        """Test getting available difficulties"""
        difficulties = self.db.get_difficulties()
        assert len(difficulties) > 0
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties
    
    def test_save_and_load_file(self):
        """Test saving and loading from file"""
        test_file = "test_samples.json"
        
        # Save to file
        self.db.save_to_file(test_file)
        assert os.path.exists(test_file)
        
        # Load from file
        new_db = MockDatabase()
        new_db.load_from_file(test_file)
        assert len(new_db.samples) == len(self.db.samples)
        
        # Clean up
        os.remove(test_file)

class TestPOSVisualizer:
    """Test cases for POSVisualizer class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.visualizer = POSVisualizer()
        self.tagger = AdvancedPOSTagger()
        self.sample_text = "The quick brown fox jumps over the lazy dog."
        self.tags = self.tagger.tag_text(self.sample_text)
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization"""
        assert self.visualizer is not None
    
    def test_create_pos_distribution_chart(self):
        """Test creating POS distribution chart"""
        fig = self.visualizer.create_pos_distribution_chart(self.tags)
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_create_pos_bar_chart(self):
        """Test creating POS bar chart"""
        fig = self.visualizer.create_pos_bar_chart(self.tags)
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_create_confidence_heatmap(self):
        """Test creating confidence heatmap"""
        fig = self.visualizer.create_confidence_heatmap(self.tags)
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_create_word_length_analysis(self):
        """Test creating word length analysis"""
        fig = self.visualizer.create_word_length_analysis(self.tags)
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_create_comprehensive_dashboard(self):
        """Test creating comprehensive dashboard"""
        fig = self.visualizer.create_comprehensive_dashboard(self.tags)
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_export_visualization_data(self):
        """Test exporting visualization data"""
        test_file = "test_viz_data.json"
        self.visualizer.export_visualization_data(self.tags, test_file)
        
        assert os.path.exists(test_file)
        
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        assert "pos_counts" in data
        assert "pos_confidences" in data
        assert "word_lengths" in data
        assert "statistics" in data
        
        # Clean up
        os.remove(test_file)

class TestPOSTag:
    """Test cases for POSTag dataclass"""
    
    def test_postag_creation(self):
        """Test POSTag object creation"""
        tag = POSTag(
            word="test",
            pos="NOUN",
            tag="NN",
            lemma="test",
            is_punct=False,
            is_space=False,
            is_stop=False,
            dep="ROOT",
            confidence=0.95,
            explanation="A noun"
        )
        
        assert tag.word == "test"
        assert tag.pos == "NOUN"
        assert tag.confidence == 0.95
        assert tag.explanation == "A noun"
    
    def test_postag_without_confidence(self):
        """Test POSTag creation without confidence"""
        tag = POSTag(
            word="test",
            pos="NOUN",
            tag="NN",
            lemma="test",
            is_punct=False,
            is_space=False,
            is_stop=False,
            dep="ROOT"
        )
        
        assert tag.confidence is None
        assert tag.explanation is None

class TestIntegration:
    """Integration tests for the entire system"""
    
    def test_end_to_end_tagging(self):
        """Test complete end-to-end tagging workflow"""
        tagger = AdvancedPOSTagger()
        visualizer = POSVisualizer()
        
        text = "The quick brown fox jumps over the lazy dog near the riverbank."
        
        # Tag text
        tags = tagger.tag_text(text, include_confidence=True)
        assert len(tags) > 0
        
        # Get statistics
        stats = tagger.get_pos_statistics(tags)
        assert stats["total_words"] > 0
        
        # Extract phrases
        phrases = tagger.extract_phrases(text)
        assert len(phrases) > 0
        
        # Analyze structure
        structure = tagger.analyze_sentence_structure(text)
        assert structure["root"] is not None
        
        # Create visualizations
        fig = visualizer.create_comprehensive_dashboard(tags)
        assert fig is not None
    
    def test_database_integration(self):
        """Test integration with mock database"""
        db = MockDatabase()
        tagger = AdvancedPOSTagger()
        
        # Get a sample text
        samples = db.get_samples_by_domain("news")
        assert len(samples) > 0
        
        sample_text = samples[0]["text"]
        
        # Tag the sample text
        tags = tagger.tag_text(sample_text)
        assert len(tags) > 0
        
        # Verify we get reasonable results
        pos_tags = [tag.pos for tag in tags if not tag.is_space and not tag.is_punct]
        assert len(pos_tags) > 0

# Fixtures for pytest
@pytest.fixture
def sample_tagger():
    """Fixture providing a sample tagger"""
    return AdvancedPOSTagger(language="en")

@pytest.fixture
def sample_text():
    """Fixture providing sample text"""
    return "The quick brown fox jumps over the lazy dog near the riverbank."

@pytest.fixture
def sample_tags(sample_tagger, sample_text):
    """Fixture providing sample tags"""
    return sample_tagger.tag_text(sample_text)

@pytest.fixture
def sample_database():
    """Fixture providing sample database"""
    return MockDatabase()

# Performance tests
class TestPerformance:
    """Performance tests for the system"""
    
    def test_large_text_performance(self):
        """Test performance with large text"""
        tagger = AdvancedPOSTagger()
        
        # Create a large text by repeating the sample
        large_text = "The quick brown fox jumps over the lazy dog near the riverbank. " * 100
        
        import time
        start_time = time.time()
        tags = tagger.tag_text(large_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 10.0  # Should process in under 10 seconds
        assert len(tags) > 0
    
    def test_batch_processing_performance(self):
        """Test performance of batch processing"""
        tagger = AdvancedPOSTagger()
        db = MockDatabase()
        
        # Get multiple samples
        samples = db.get_all_samples()[:10]  # First 10 samples
        
        import time
        start_time = time.time()
        
        for sample in samples:
            tags = tagger.tag_text(sample["text"])
            assert len(tags) > 0
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 10 samples in reasonable time
        assert processing_time < 30.0

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
