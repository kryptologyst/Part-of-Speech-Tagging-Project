# Part-of-Speech Tagging Project

A modern, comprehensive Part-of-Speech (POS) tagging application with web interface, visualization, and advanced NLP features.

## Features

- **Advanced POS Tagging**: Using latest spaCy models with confidence scores
- **Multi-language Support**: English, Spanish, French, German, and more
- **Web Interface**: Interactive Flask-based UI for real-time tagging
- **Visualization**: POS tag distribution charts and sentence structure diagrams
- **Mock Database**: Sample texts from various domains (news, literature, technical)
- **API Endpoints**: RESTful API for programmatic access
- **Custom Models**: Support for fine-tuned models and custom tagging rules
- **Comprehensive Testing**: Full test suite with pytest

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/0106_Part-of-speech_tagging.git
cd 0106_Part-of-speech_tagging
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download spaCy models:**
```bash
# English (required)
python -m spacy download en_core_web_sm

# Optional: Other languages
python -m spacy download es_core_news_sm  # Spanish
python -m spacy download fr_core_news_sm  # French
python -m spacy download de_core_news_sm  # German
python -m spacy download it_core_news_sm  # Italian
python -m spacy download pt_core_news_sm  # Portuguese
python -m spacy download nl_core_news_sm  # Dutch
python -m spacy download ru_core_news_sm  # Russian
python -m spacy download zh_core_web_sm   # Chinese
python -m spacy download ja_core_news_sm # Japanese
```

## Usage

### Command Line Interface

```bash
# Basic tagging
python pos_tagger.py --text "The quick brown fox jumps over the lazy dog."

# With additional options
python pos_tagger.py --text "Your text here" --language es --stats --phrases --structure --output results.json

# Help
python pos_tagger.py --help
```

### Web Interface

1. **Start the Flask application:**
```bash
python app.py
```

2. **Open your browser and navigate to:**
```
http://localhost:5000
```

3. **Use the interactive interface to:**
   - Enter text for POS tagging
   - Select language and options
   - View results with visualizations
   - Browse sample texts
   - Analyze sentence structure

### API Usage

#### Tag Text
```bash
curl -X POST http://localhost:5000/api/tag \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "language": "en",
    "include_stats": true,
    "include_phrases": true,
    "include_structure": true
  }'
```

#### Get Sample Texts
```bash
# Get all samples
curl http://localhost:5000/api/samples

# Filter by domain
curl http://localhost:5000/api/samples?domain=news

# Search samples
curl http://localhost:5000/api/samples?search=AI
```

#### Batch Processing
```bash
curl -X POST http://localhost:5000/api/batch_tag \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "First sentence to analyze.",
      "Second sentence to analyze."
    ],
    "language": "en",
    "include_stats": true
  }'
```

### Python API

```python
from pos_tagger import AdvancedPOSTagger
from utils.visualizer import POSVisualizer

# Initialize tagger
tagger = AdvancedPOSTagger(language="en")

# Tag text
text = "The quick brown fox jumps over the lazy dog."
tags = tagger.tag_text(text, include_confidence=True)

# Get statistics
stats = tagger.get_pos_statistics(tags)
print(f"Total words: {stats['total_words']}")
print(f"POS distribution: {stats['pos_percentages']}")

# Extract phrases
phrases = tagger.extract_phrases(text)
print(f"Noun phrases: {[p for p in phrases if p['type'] == 'noun_phrase']}")

# Create visualizations
visualizer = POSVisualizer()
fig = visualizer.create_comprehensive_dashboard(tags)
fig.show()
```

## Project Structure

```
├── app.py                    # Flask web application
├── pos_tagger.py             # Core POS tagging functionality
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
├── .gitignore               # Git ignore rules
├── .github/                  # GitHub workflows and templates
│   ├── workflows/
│   │   ├── ci.yml           # Continuous Integration
│   │   └── deploy.yml       # Deployment workflow
│   └── ISSUE_TEMPLATE/       # Issue templates
├── data/                     # Data and mock database
│   ├── mock_database.py     # Sample texts database
│   └── sample_texts.json    # Exported sample texts
├── static/                   # Static web assets
│   ├── css/                 # CSS stylesheets
│   └── js/                  # JavaScript files
├── templates/               # HTML templates
│   └── index.html           # Main web interface
├── tests/                   # Test suite
│   ├── test_pos_tagger.py   # Main test file
│   └── conftest.py          # Test configuration
├── utils/                   # Utility modules
│   ├── visualizer.py        # Visualization utilities
│   └── helpers.py           # Helper functions
└── models/                  # Custom models (future)
    └── README.md            # Model documentation
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pos_tagger --cov=utils --cov=data

# Run specific test file
pytest tests/test_pos_tagger.py -v

# Run with detailed output
pytest tests/ -v -s
```

## Visualization

The project includes comprehensive visualization capabilities:

- **POS Distribution Charts**: Pie charts and bar charts showing tag frequencies
- **Confidence Heatmaps**: Visual representation of model confidence
- **Dependency Trees**: Sentence structure visualization
- **Word Length Analysis**: Statistical analysis of word lengths by POS
- **Comprehensive Dashboard**: Multi-panel visualization combining all charts

## Supported Languages

- **English** (en) - Primary language
- **Spanish** (es) - Español
- **French** (fr) - Français
- **German** (de) - Deutsch
- **Italian** (it) - Italiano
- **Portuguese** (pt) - Português
- **Dutch** (nl) - Nederlands
- **Russian** (ru) - Русский
- **Chinese** (zh) - 中文
- **Japanese** (ja) - 日本語

## 🔧 Configuration

### Environment Variables

Create a `.env` file for configuration:

```bash
# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# spaCy configuration
SPACY_MODEL_PATH=models/
SPACY_CACHE_DIR=cache/

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Custom Models

To use custom spaCy models:

1. Place your model in the `models/` directory
2. Update the `supported_languages` dictionary in `pos_tagger.py`
3. Restart the application

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t pos-tagger .

# Run container
docker run -p 5000:5000 pos-tagger
```

### Production Deployment

For production deployment:

1. Set `FLASK_ENV=production`
2. Use a production WSGI server (e.g., Gunicorn)
3. Configure reverse proxy (e.g., Nginx)
4. Set up SSL certificates
5. Configure monitoring and logging

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run tests: `pytest tests/`
6. Commit changes: `git commit -m "Add feature"`
7. Push to branch: `git push origin feature-name`
8. Submit a pull request

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **Pytest** for testing
- **Type hints** for better code documentation

```bash
# Format code
black .

# Lint code
flake8 .

# Run tests
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **spaCy** team for the excellent NLP library
- **Flask** team for the web framework
- **Plotly** team for visualization capabilities
- **Bootstrap** team for the UI framework
- **Font Awesome** for the icons


## Future Enhancements

- [ ] Custom model training interface
- [ ] Real-time collaborative tagging
- [ ] Advanced dependency parsing visualization
- [ ] Integration with more NLP libraries
- [ ] Mobile app development
- [ ] Cloud deployment options
- [ ] Performance optimization
- [ ] Additional language support


# Part-of-Speech-Tagging-Project
