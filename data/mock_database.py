"""
Mock Database for Part-of-Speech Tagging
Contains sample texts from various domains for testing and demonstration
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SampleText:
    """Data class for sample text entries"""
    id: str
    title: str
    text: str
    domain: str
    language: str
    difficulty: str
    description: str

class MockDatabase:
    """Mock database containing sample texts for POS tagging"""
    
    def __init__(self):
        self.samples = self._load_sample_texts()
    
    def _load_sample_texts(self) -> List[SampleText]:
        """Load sample texts from various domains"""
        samples = [
            # News samples
            SampleText(
                id="news_001",
                title="Technology Breakthrough",
                text="Scientists have developed a revolutionary artificial intelligence system that can process natural language with unprecedented accuracy. The breakthrough promises to transform how computers understand human communication.",
                domain="news",
                language="en",
                difficulty="medium",
                description="Technology news article with technical terms"
            ),
            SampleText(
                id="news_002", 
                title="Economic Update",
                text="The Federal Reserve announced today that interest rates will remain unchanged for the foreseeable future. Economists predict this decision will stabilize markets and encourage continued economic growth.",
                domain="news",
                language="en",
                difficulty="medium",
                description="Financial news with economic terminology"
            ),
            
            # Literature samples
            SampleText(
                id="lit_001",
                title="Classic Poetry",
                text="The quick brown fox jumps over the lazy dog near the riverbank. This sentence contains every letter of the alphabet and demonstrates various parts of speech beautifully.",
                domain="literature",
                language="en",
                difficulty="easy",
                description="Famous pangram sentence"
            ),
            SampleText(
                id="lit_002",
                title="Shakespeare Excerpt",
                text="To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
                domain="literature", 
                language="en",
                difficulty="hard",
                description="Shakespearean English with archaic forms"
            ),
            
            # Technical samples
            SampleText(
                id="tech_001",
                title="Programming Documentation",
                text="The function initialize_model() loads the pre-trained weights and configures the neural network architecture. It returns a model object that can be used for inference or fine-tuning.",
                domain="technical",
                language="en",
                difficulty="hard",
                description="Programming documentation with technical jargon"
            ),
            SampleText(
                id="tech_002",
                title="API Reference",
                text="POST /api/v1/tag - Submit text for part-of-speech tagging. The request body should contain a JSON object with 'text' and optional 'language' fields.",
                domain="technical",
                language="en", 
                difficulty="medium",
                description="API documentation format"
            ),
            
            # Conversational samples
            SampleText(
                id="conv_001",
                title="Casual Conversation",
                text="Hey, how are you doing today? I was thinking we could grab some coffee later if you're free. What do you think?",
                domain="conversational",
                language="en",
                difficulty="easy",
                description="Informal conversational text"
            ),
            SampleText(
                id="conv_002",
                title="Formal Email",
                text="Dear Professor Smith, I hope this email finds you well. I am writing to inquire about the research opportunities available in your laboratory this summer.",
                domain="conversational",
                language="en",
                difficulty="medium", 
                description="Formal written communication"
            ),
            
            # Scientific samples
            SampleText(
                id="sci_001",
                title="Research Abstract",
                text="This study investigates the effectiveness of machine learning algorithms in natural language processing tasks. Our results demonstrate significant improvements in accuracy compared to traditional statistical methods.",
                domain="scientific",
                language="en",
                difficulty="hard",
                description="Academic research writing"
            ),
            SampleText(
                id="sci_002",
                title="Medical Report",
                text="The patient presented with acute symptoms including fever, headache, and fatigue. Laboratory tests revealed elevated white blood cell count, suggesting possible infection.",
                domain="scientific",
                language="en",
                difficulty="hard",
                description="Medical terminology and clinical language"
            ),
            
            # Multilingual samples
            SampleText(
                id="multi_001",
                title="Spanish Sample",
                text="El rápido zorro marrón salta sobre el perro perezoso cerca del río. Esta oración contiene muchas palabras diferentes y es útil para probar el etiquetado morfológico.",
                domain="multilingual",
                language="es",
                difficulty="medium",
                description="Spanish text for multilingual testing"
            ),
            SampleText(
                id="multi_002",
                title="French Sample", 
                text="Le renard brun rapide saute par-dessus le chien paresseux près de la rivière. Cette phrase contient plusieurs types de mots et est parfaite pour l'analyse grammaticale.",
                domain="multilingual",
                language="fr",
                difficulty="medium",
                description="French text for multilingual testing"
            )
        ]
        return samples
    
    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Get all sample texts as dictionaries"""
        return [
            {
                "id": sample.id,
                "title": sample.title,
                "text": sample.text,
                "domain": sample.domain,
                "language": sample.language,
                "difficulty": sample.difficulty,
                "description": sample.description
            }
            for sample in self.samples
        ]
    
    def get_samples_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get samples filtered by domain"""
        return [
            {
                "id": sample.id,
                "title": sample.title,
                "text": sample.text,
                "domain": sample.domain,
                "language": sample.language,
                "difficulty": sample.difficulty,
                "description": sample.description
            }
            for sample in self.samples
            if sample.domain == domain
        ]
    
    def get_samples_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Get samples filtered by language"""
        return [
            {
                "id": sample.id,
                "title": sample.title,
                "text": sample.text,
                "domain": sample.domain,
                "language": sample.language,
                "difficulty": sample.difficulty,
                "description": sample.description
            }
            for sample in self.samples
            if sample.language == language
        ]
    
    def get_sample_by_id(self, sample_id: str) -> Dict[str, Any]:
        """Get a specific sample by ID"""
        for sample in self.samples:
            if sample.id == sample_id:
                return {
                    "id": sample.id,
                    "title": sample.title,
                    "text": sample.text,
                    "domain": sample.domain,
                    "language": sample.language,
                    "difficulty": sample.difficulty,
                    "description": sample.description
                }
        return None
    
    def get_domains(self) -> List[str]:
        """Get list of available domains"""
        return list(set(sample.domain for sample in self.samples))
    
    def get_languages(self) -> List[str]:
        """Get list of available languages"""
        return list(set(sample.language for sample in self.samples))
    
    def get_difficulties(self) -> List[str]:
        """Get list of available difficulty levels"""
        return list(set(sample.difficulty for sample in self.samples))
    
    def search_samples(self, query: str) -> List[Dict[str, Any]]:
        """Search samples by title or description"""
        query_lower = query.lower()
        results = []
        
        for sample in self.samples:
            if (query_lower in sample.title.lower() or 
                query_lower in sample.description.lower() or
                query_lower in sample.text.lower()):
                results.append({
                    "id": sample.id,
                    "title": sample.title,
                    "text": sample.text,
                    "domain": sample.domain,
                    "language": sample.language,
                    "difficulty": sample.difficulty,
                    "description": sample.description
                })
        
        return results
    
    def save_to_file(self, filename: str):
        """Save all samples to a JSON file"""
        data = {
            "samples": self.get_all_samples(),
            "metadata": {
                "total_samples": len(self.samples),
                "domains": self.get_domains(),
                "languages": self.get_languages(),
                "difficulties": self.get_difficulties()
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filename: str):
        """Load samples from a JSON file"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = [
            SampleText(
                id=sample["id"],
                title=sample["title"],
                text=sample["text"],
                domain=sample["domain"],
                language=sample["language"],
                difficulty=sample["difficulty"],
                description=sample["description"]
            )
            for sample in data["samples"]
        ]

# Global instance for easy access
mock_db = MockDatabase()

if __name__ == "__main__":
    # Demo usage
    db = MockDatabase()
    
    print("Available domains:", db.get_domains())
    print("Available languages:", db.get_languages())
    print("Available difficulties:", db.get_difficulties())
    print("\nSample news articles:")
    
    news_samples = db.get_samples_by_domain("news")
    for sample in news_samples:
        print(f"- {sample['title']}: {sample['description']}")
    
    print("\nSearching for 'AI':")
    ai_samples = db.search_samples("AI")
    for sample in ai_samples:
        print(f"- {sample['title']}: {sample['text'][:50]}...")
    
    # Save to file
    db.save_to_file("sample_texts.json")
    print("\nSample texts saved to sample_texts.json")
