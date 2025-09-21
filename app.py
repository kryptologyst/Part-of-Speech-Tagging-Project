#!/usr/bin/env python3
"""
Flask Web Application for Part-of-Speech Tagging
Provides interactive UI and REST API for POS tagging
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from pos_tagger import AdvancedPOSTagger
from data.mock_database import MockDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
tagger = AdvancedPOSTagger()
mock_db = MockDatabase()

@app.route('/')
def index():
    """Main page with interactive POS tagging interface"""
    return render_template('index.html')

@app.route('/api/tag', methods=['POST'])
def tag_text():
    """API endpoint for POS tagging"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        language = data.get('language', 'en')
        include_stats = data.get('include_stats', False)
        include_phrases = data.get('include_phrases', False)
        include_structure = data.get('include_structure', False)
        
        # Update tagger language if different
        if language != tagger.language:
            tagger.language = language
            tagger._load_model()
        
        # Tag the text
        tags = tagger.tag_text(text, include_confidence=True)
        
        # Prepare response
        response = {
            'text': text,
            'language': language,
            'tags': [
                {
                    'word': tag.word,
                    'pos': tag.pos,
                    'tag': tag.tag,
                    'lemma': tag.lemma,
                    'is_punct': tag.is_punct,
                    'is_space': tag.is_space,
                    'is_stop': tag.is_stop,
                    'dep': tag.dep,
                    'confidence': tag.confidence,
                    'explanation': tag.explanation
                }
                for tag in tags
            ]
        }
        
        if include_stats:
            response['statistics'] = tagger.get_pos_statistics(tags)
        
        if include_phrases:
            response['phrases'] = tagger.extract_phrases(text)
        
        if include_structure:
            response['structure'] = tagger.analyze_sentence_structure(text)
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in tag_text: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/samples')
def get_samples():
    """Get sample texts from mock database"""
    try:
        domain = request.args.get('domain')
        language = request.args.get('language')
        difficulty = request.args.get('difficulty')
        search = request.args.get('search')
        
        if search:
            samples = mock_db.search_samples(search)
        elif domain:
            samples = mock_db.get_samples_by_domain(domain)
        elif language:
            samples = mock_db.get_samples_by_language(language)
        else:
            samples = mock_db.get_all_samples()
        
        # Filter by difficulty if specified
        if difficulty:
            samples = [s for s in samples if s['difficulty'] == difficulty]
        
        return jsonify({
            'samples': samples,
            'total': len(samples)
        })
    
    except Exception as e:
        logger.error(f"Error in get_samples: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/samples/<sample_id>')
def get_sample(sample_id):
    """Get a specific sample by ID"""
    try:
        sample = mock_db.get_sample_by_id(sample_id)
        if sample:
            return jsonify(sample)
        else:
            return jsonify({'error': 'Sample not found'}), 404
    
    except Exception as e:
        logger.error(f"Error in get_sample: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metadata')
def get_metadata():
    """Get metadata about available options"""
    try:
        return jsonify({
            'domains': mock_db.get_domains(),
            'languages': mock_db.get_languages(),
            'difficulties': mock_db.get_difficulties(),
            'supported_languages': tagger.supported_languages
        })
    
    except Exception as e:
        logger.error(f"Error in get_metadata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_tag', methods=['POST'])
def batch_tag():
    """Tag multiple texts in batch"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Texts array is required'}), 400
        
        texts = data['texts']
        language = data.get('language', 'en')
        include_stats = data.get('include_stats', False)
        
        # Update tagger language if different
        if language != tagger.language:
            tagger.language = language
            tagger._load_model()
        
        results = []
        for text_data in texts:
            if isinstance(text_data, str):
                text = text_data
            elif isinstance(text_data, dict) and 'text' in text_data:
                text = text_data['text']
            else:
                continue
            
            tags = tagger.tag_text(text, include_confidence=True)
            
            result = {
                'text': text,
                'tags': [
                    {
                        'word': tag.word,
                        'pos': tag.pos,
                        'tag': tag.tag,
                        'lemma': tag.lemma,
                        'is_punct': tag.is_punct,
                        'is_space': tag.is_space,
                        'is_stop': tag.is_stop,
                        'dep': tag.dep,
                        'confidence': tag.confidence,
                        'explanation': tag.explanation
                    }
                    for tag in tags
                ]
            }
            
            if include_stats:
                result['statistics'] = tagger.get_pos_statistics(tags)
            
            results.append(result)
        
        return jsonify({
            'results': results,
            'total': len(results)
        })
    
    except Exception as e:
        logger.error(f"Error in batch_tag: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'tagger_language': tagger.language,
        'supported_languages': list(tagger.supported_languages.keys())
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
