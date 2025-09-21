import os
import json
import hashlib
import numpy as np
import networkx as nx
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import cv2
from PIL import Image, ImageChops, ImageEnhance
import re
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# In-memory storage for prototype (use database in production)
content_graph = nx.DiGraph()
content_store = {}

class ContentAnalyzer:
    def __init__(self):
        self.mutation_threshold = 0.7
    
    def generate_content_hash(self, content, content_type):
        """Generate unique fingerprint for content"""
        if content_type == 'text':
            # Normalize text for hashing
            normalized = re.sub(r'\s+', ' ', content.lower().strip())
            return hashlib.md5(normalized.encode()).hexdigest()[:12]
        elif content_type == 'image':
            # Simple image hash based on histogram
            img_array = np.array(content)
            hist = cv2.calcHist([img_array], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            return hashlib.md5(hist.tobytes()).hexdigest()[:12]
    
    def analyze_text_mutation(self, original, current):
        """Analyze how text has mutated"""
        original_words = set(original.lower().split())
        current_words = set(current.lower().split())
        
        # Calculate similarity metrics
        intersection = len(original_words.intersection(current_words))
        union = len(original_words.union(current_words))
        jaccard_similarity = intersection / union if union > 0 else 0
        
        # Analyze sentiment/tone changes
        original_caps = sum(1 for c in original if c.isupper())
        current_caps = sum(1 for c in current if c.isupper())
        tone_change = abs(original_caps - current_caps) / max(len(original), 1)
        
        return {
            'similarity': jaccard_similarity,
            'tone_change': tone_change,
            'word_changes': {
                'added': list(current_words - original_words),
                'removed': list(original_words - current_words)
            },
            'mutation_score': 1 - jaccard_similarity + tone_change
        }
    
    def analyze_image_manipulation(self, image):
        """Detect potential image manipulation"""
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Error Level Analysis (simplified)
        resaved = image.copy()
        resaved.save(BytesIO(), 'JPEG', quality=95)
        
        # Check for uniform regions (potential manipulation)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Analyze color distribution
        hist_r = cv2.calcHist([img_array], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_array], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_array], [2], None, [256], [0, 256])
        
        # Simple manipulation score based on unusual color distributions
        color_uniformity = np.std([np.std(hist_r), np.std(hist_g), np.std(hist_b)])
        
        return {
            'edge_density': float(edge_density),
            'color_uniformity': float(color_uniformity),
            'manipulation_score': float(1 - edge_density + color_uniformity / 1000),
            'suspicious_regions': edge_density < 0.1 or color_uniformity > 1000
        }

class MutationTracker:
    def __init__(self):
        self.analyzer = ContentAnalyzer()
    
    def add_content(self, content, content_type, platform, user_id=None, parent_hash=None):
        """Add new content to the mutation graph"""
        content_hash = self.analyzer.generate_content_hash(content, content_type)
        timestamp = datetime.now().isoformat()
        
        # Store content data
        content_data = {
            'hash': content_hash,
            'content': content if content_type == 'text' else 'binary_data',
            'type': content_type,
            'platform': platform,
            'timestamp': timestamp,
            'user_id': user_id,
            'analysis': {}
        }
        
        # Perform analysis based on content type
        if content_type == 'text':
            # Check for similar existing content
            parent_hash = self.find_text_parent(content)
            if parent_hash:
                original_content = content_store[parent_hash]['content']
                content_data['analysis'] = self.analyzer.analyze_text_mutation(original_content, content)
        elif content_type == 'image':
            content_data['analysis'] = self.analyzer.analyze_image_manipulation(content)
        
        content_store[content_hash] = content_data
        
        # Add to graph
        content_graph.add_node(content_hash, **content_data)
        
        # Add edge if parent exists
        if parent_hash and parent_hash in content_store:
            mutation_score = content_data['analysis'].get('mutation_score', 0)
            content_graph.add_edge(parent_hash, content_hash, 
                                 weight=mutation_score,
                                 platform_transition=f"{content_store[parent_hash]['platform']} → {platform}")
        
        return content_hash
    
    def find_text_parent(self, text):
        """Find most similar existing text content"""
        best_match = None
        best_similarity = 0
        
        for hash_id, data in content_store.items():
            if data['type'] == 'text':
                analysis = self.analyzer.analyze_text_mutation(data['content'], text)
                if analysis['similarity'] > best_similarity and analysis['similarity'] > 0.5:
                    best_similarity = analysis['similarity']
                    best_match = hash_id
        
        return best_match
    
    def get_mutation_path(self, content_hash):
        """Get the complete mutation path for a piece of content"""
        if content_hash not in content_graph:
            return []
        
        # Find root nodes (sources)
        predecessors = list(content_graph.predecessors(content_hash))
        if not predecessors:
            return [content_hash]
        
        # Trace back to find the path
        path = [content_hash]
        current = content_hash
        
        while predecessors:
            # Get the most likely parent (highest similarity)
            parent = predecessors[0]  # Simplified - take first predecessor
            path.insert(0, parent)
            current = parent
            predecessors = list(content_graph.predecessors(current))
        
        return path
    
    def generate_mutation_map(self, content_hash):
        """Generate visualization data for mutation map"""
        path = self.get_mutation_path(content_hash)
        
        mutation_data = []
        for i, hash_id in enumerate(path):
            data = content_store[hash_id]
            mutation_info = {
                'step': i,
                'hash': hash_id,
                'platform': data['platform'],
                'timestamp': data['timestamp'],
                'analysis': data.get('analysis', {}),
                'content_preview': data['content'][:100] if data['type'] == 'text' else 'Image data'
            }
            mutation_data.append(mutation_info)
        
        return mutation_data

# Initialize tracker
tracker = MutationTracker()

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Misinformation DNA Fingerprinting</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        .header { text-align: center; margin-bottom: 40px; }
        .upload-section { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
        .results { margin-top: 30px; }
        .mutation-step { background: #e3f2fd; padding: 15px; margin: 10px 0; border-left: 4px solid #2196f3; }
        .analysis-box { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }
        button { background: #2196f3; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #1976d2; }
        input, select, textarea { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: #f1f8e9; padding: 15px; border-radius: 8px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Misinformation DNA Fingerprinting</h1>
            <p>Track how information mutates as it spreads across platforms</p>
        </div>
        
        <div class="upload-section">
            <h2>Submit Content for Analysis</h2>
            <form id="contentForm" enctype="multipart/form-data">
                <div>
                    <label>Content Type:</label>
                    <select id="contentType" onchange="toggleInput()">
                        <option value="text">Text</option>
                        <option value="image">Image</option>
                    </select>
                </div>
                <div>
                    <label>Platform:</label>
                    <select id="platform">
                        <option value="twitter">Twitter</option>
                        <option value="facebook">Facebook</option>
                        <option value="whatsapp">WhatsApp</option>
                        <option value="telegram">Telegram</option>
                        <option value="other">Other</option>
                    </select>
                </div>
                <div id="textInput">
                    <label>Text Content:</label><br>
                    <textarea id="textContent" rows="4" cols="50" placeholder="Enter the text content here..."></textarea>
                </div>
                <div id="imageInput" style="display:none;">
                    <label>Image File:</label><br>
                    <input type="file" id="imageFile" accept="image/*">
                </div>
                <div>
                    <button type="button" onclick="submitContent()">Analyze Content</button>
                </div>
            </form>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3 id="totalContent">0</h3>
                <p>Total Content Analyzed</p>
            </div>
            <div class="stat-card">
                <h3 id="mutationChains">0</h3>
                <p>Mutation Chains Detected</p>
            </div>
            <div class="stat-card">
                <h3 id="platforms">0</h3>
                <p>Platforms Tracked</p>
            </div>
        </div>
        
        <div id="results" class="results"></div>
    </div>
    
    <script>
        function toggleInput() {
            const type = document.getElementById('contentType').value;
            const textInput = document.getElementById('textInput');
            const imageInput = document.getElementById('imageInput');
            
            if (type === 'text') {
                textInput.style.display = 'block';
                imageInput.style.display = 'none';
            } else {
                textInput.style.display = 'none';
                imageInput.style.display = 'block';
            }
        }
        
        async function submitContent() {
            const contentType = document.getElementById('contentType').value;
            const platform = document.getElementById('platform').value;
            const formData = new FormData();
            
            formData.append('content_type', contentType);
            formData.append('platform', platform);
            
            if (contentType === 'text') {
                const textContent = document.getElementById('textContent').value;
                if (!textContent.trim()) {
                    alert('Please enter text content');
                    return;
                }
                formData.append('content', textContent);
            } else {
                const imageFile = document.getElementById('imageFile').files[0];
                if (!imageFile) {
                    alert('Please select an image file');
                    return;
                }
                formData.append('file', imageFile);
            }
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                displayResults(result);
                updateStats();
                
            } catch (error) {
                console.error('Error:', error);
                alert('Error analyzing content');
            }
        }
        
        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            
            let html = '<h2>🔍 Analysis Results</h2>';
            html += `<div class="analysis-box">
                <h3>Content Hash: ${result.hash}</h3>
                <p><strong>Platform:</strong> ${result.platform}</p>
                <p><strong>Timestamp:</strong> ${result.timestamp}</p>
            </div>`;
            
            if (result.mutation_map && result.mutation_map.length > 1) {
                html += '<h3>📊 Mutation Map</h3>';
                result.mutation_map.forEach((step, index) => {
                    html += `<div class="mutation-step">
                        <h4>Step ${step.step + 1}: ${step.platform}</h4>
                        <p><strong>Time:</strong> ${step.timestamp}</p>
                        <p><strong>Content Preview:</strong> ${step.content_preview}</p>`;
                    
                    if (step.analysis.similarity !== undefined) {
                        html += `<p><strong>Similarity to Original:</strong> ${(step.analysis.similarity * 100).toFixed(1)}%</p>`;
                        html += `<p><strong>Mutation Score:</strong> ${(step.analysis.mutation_score * 100).toFixed(1)}%</p>`;
                    }
                    
                    if (step.analysis.manipulation_score !== undefined) {
                        html += `<p><strong>Manipulation Score:</strong> ${(step.analysis.manipulation_score * 100).toFixed(1)}%</p>`;
                        html += `<p><strong>Suspicious Regions:</strong> ${step.analysis.suspicious_regions ? 'Yes' : 'No'}</p>`;
                    }
                    
                    html += '</div>';
                });
            } else {
                html += '<div class="analysis-box"><p>ℹ️ This appears to be original content or no mutations detected.</p></div>';
            }
            
            if (result.analysis) {
                html += '<h3>📈 Detailed Analysis</h3>';
                html += `<div class="analysis-box">
                    <pre>${JSON.stringify(result.analysis, null, 2)}</pre>
                </div>`;
            }
            
            resultsDiv.innerHTML = html;
        }
        
        async function updateStats() {
            try {
                const response = await fetch('/stats');
                const stats = await response.json();
                
                document.getElementById('totalContent').textContent = stats.total_content;
                document.getElementById('mutationChains').textContent = stats.mutation_chains;
                document.getElementById('platforms').textContent = stats.platforms_count;
                
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }
        
        // Load initial stats
        updateStats();
    </script>
</body>
</html>
    ''')

@app.route('/analyze', methods=['POST'])
def analyze_content():
    try:
        content_type = request.form.get('content_type')
        platform = request.form.get('platform')
        
        if content_type == 'text':
            content = request.form.get('content')
            if not content:
                return jsonify({'error': 'No content provided'}), 400
        elif content_type == 'image':
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Process image
            image = Image.open(file.stream)
            content = image
        else:
            return jsonify({'error': 'Invalid content type'}), 400
        
        # Add content to tracker
        content_hash = tracker.add_content(content, content_type, platform)
        
        # Get mutation map
        mutation_map = tracker.generate_mutation_map(content_hash)
        
        # Prepare response
        response_data = {
            'hash': content_hash,
            'platform': platform,
            'timestamp': content_store[content_hash]['timestamp'],
            'mutation_map': mutation_map,
            'analysis': content_store[content_hash]['analysis']
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    platforms = set()
    mutation_chains = 0
    
    for data in content_store.values():
        platforms.add(data['platform'])
    
    # Count mutation chains (nodes with predecessors)
    for node in content_graph.nodes():
        if list(content_graph.predecessors(node)):
            mutation_chains += 1
    
    return jsonify({
        'total_content': len(content_store),
        'mutation_chains': mutation_chains,
        'platforms_count': len(platforms),
        'platforms': list(platforms)
    })

@app.route('/mutation-map/<content_hash>')
def get_mutation_map(content_hash):
    if content_hash not in content_store:
        return jsonify({'error': 'Content not found'}), 404
    
    mutation_map = tracker.generate_mutation_map(content_hash)
    return jsonify({'mutation_map': mutation_map})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'version': '1.0.0'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)