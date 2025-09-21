#!/usr/bin/env python3
"""
Advanced Part-of-Speech Tagging Module
Supports multiple languages, confidence scores, and custom models
"""

import spacy
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class POSTag:
    """Data class for POS tag information"""
    word: str
    pos: str
    tag: str
    lemma: str
    is_punct: bool
    is_space: bool
    is_stop: bool
    dep: str
    confidence: Optional[float] = None
    explanation: Optional[str] = None

class AdvancedPOSTagger:
    """Advanced POS tagger with multiple language support and confidence scoring"""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.nlp = None
        self.supported_languages = {
            "en": "en_core_web_sm",
            "es": "es_core_news_sm", 
            "fr": "fr_core_news_sm",
            "de": "de_core_news_sm",
            "it": "it_core_news_sm",
            "pt": "pt_core_news_sm",
            "nl": "nl_core_news_sm",
            "ru": "ru_core_news_sm",
            "zh": "zh_core_web_sm",
            "ja": "ja_core_news_sm"
        }
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate spaCy model for the language"""
        try:
            model_name = self.supported_languages.get(self.language, "en_core_web_sm")
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded {model_name} model for language: {self.language}")
        except OSError:
            logger.warning(f"Model {model_name} not found. Please install it with: python -m spacy download {model_name}")
            # Fallback to English
            self.nlp = spacy.load("en_core_web_sm")
            self.language = "en"
    
    def tag_text(self, text: str, include_confidence: bool = True) -> List[POSTag]:
        """
        Tag text with POS information
        
        Args:
            text: Input text to tag
            include_confidence: Whether to include confidence scores
            
        Returns:
            List of POSTag objects
        """
        if not self.nlp:
            raise ValueError("No model loaded. Please check model installation.")
        
        doc = self.nlp(text)
        tags = []
        
        for token in doc:
            # Calculate confidence based on model certainty
            confidence = None
            if include_confidence:
                # Use token probability if available
                confidence = getattr(token, 'prob', 0.8)  # Default confidence
            
            tag = POSTag(
                word=token.text,
                pos=token.pos_,
                tag=token.tag_,
                lemma=token.lemma_,
                is_punct=token.is_punct,
                is_space=token.is_space,
                is_stop=token.is_stop,
                dep=token.dep_,
                confidence=confidence,
                explanation=spacy.explain(token.pos_)
            )
            tags.append(tag)
        
        return tags
    
    def get_pos_statistics(self, tags: List[POSTag]) -> Dict[str, Any]:
        """Calculate statistics from POS tags"""
        pos_counts = {}
        total_words = 0
        
        for tag in tags:
            if not tag.is_space and not tag.is_punct:
                pos_counts[tag.pos] = pos_counts.get(tag.pos, 0) + 1
                total_words += 1
        
        # Calculate percentages
        pos_percentages = {
            pos: (count / total_words) * 100 
            for pos, count in pos_counts.items()
        }
        
        return {
            "total_words": total_words,
            "pos_counts": pos_counts,
            "pos_percentages": pos_percentages,
            "unique_pos_tags": len(pos_counts)
        }
    
    def extract_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Extract noun phrases and verb phrases"""
        doc = self.nlp(text)
        phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            phrases.append({
                "text": chunk.text,
                "type": "noun_phrase",
                "root": chunk.root.text,
                "start": chunk.start,
                "end": chunk.end
            })
        
        # Extract verb phrases (simplified)
        for token in doc:
            if token.pos_ == "VERB":
                # Find the verb phrase by looking for auxiliaries and main verb
                verb_phrase = []
                for child in token.children:
                    if child.dep_ in ["aux", "auxpass"]:
                        verb_phrase.append(child.text)
                verb_phrase.append(token.text)
                
                phrases.append({
                    "text": " ".join(verb_phrase),
                    "type": "verb_phrase",
                    "root": token.text,
                    "start": token.i,
                    "end": token.i + 1
                })
        
        return phrases
    
    def analyze_sentence_structure(self, text: str) -> Dict[str, Any]:
        """Analyze sentence structure and dependencies"""
        doc = self.nlp(text)
        
        # Find root of the sentence
        root = [token for token in doc if token.dep_ == "ROOT"][0] if doc else None
        
        # Analyze dependencies
        dependencies = []
        for token in doc:
            dependencies.append({
                "word": token.text,
                "dep": token.dep_,
                "head": token.head.text,
                "pos": token.pos_
            })
        
        return {
            "root": root.text if root else None,
            "root_pos": root.pos_ if root else None,
            "dependencies": dependencies,
            "sentence_length": len(doc)
        }

def main():
    """Command line interface for POS tagging"""
    parser = argparse.ArgumentParser(description="Advanced Part-of-Speech Tagger")
    parser.add_argument("--text", "-t", required=True, help="Text to tag")
    parser.add_argument("--language", "-l", default="en", help="Language code (default: en)")
    parser.add_argument("--output", "-o", help="Output file (JSON format)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--stats", "-s", action="store_true", help="Include statistics")
    parser.add_argument("--phrases", "-p", action="store_true", help="Extract phrases")
    parser.add_argument("--structure", action="store_true", help="Analyze sentence structure")
    
    args = parser.parse_args()
    
    # Initialize tagger
    tagger = AdvancedPOSTagger(language=args.language)
    
    # Tag text
    tags = tagger.tag_text(args.text, include_confidence=True)
    
    # Prepare output
    output = {
        "text": args.text,
        "language": args.language,
        "tags": [
            {
                "word": tag.word,
                "pos": tag.pos,
                "tag": tag.tag,
                "lemma": tag.lemma,
                "is_punct": tag.is_punct,
                "is_space": tag.is_space,
                "is_stop": tag.is_stop,
                "dep": tag.dep,
                "confidence": tag.confidence,
                "explanation": tag.explanation
            }
            for tag in tags
        ]
    }
    
    if args.stats:
        output["statistics"] = tagger.get_pos_statistics(tags)
    
    if args.phrases:
        output["phrases"] = tagger.extract_phrases(args.text)
    
    if args.structure:
        output["structure"] = tagger.analyze_sentence_structure(args.text)
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
    else:
        if args.verbose:
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            # Simple table format
            print("Word\t\tPOS\t\tTag\t\tExplanation")
            print("-" * 60)
            for tag in tags:
                if not tag.is_space:
                    print(f"{tag.word:<12}\t{tag.pos:<8}\t{tag.tag:<8}\t{tag.explanation}")

if __name__ == "__main__":
    main()
