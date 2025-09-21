"""
Visualization utilities for Part-of-Speech tagging
Creates charts and diagrams for POS analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any
import json
from pos_tagger import POSTag

class POSVisualizer:
    """Create visualizations for POS tagging results"""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize the visualizer with a specific style"""
        plt.style.use(style)
        sns.set_palette("husl")
    
    def create_pos_distribution_chart(self, tags: List[POSTag], save_path: str = None) -> go.Figure:
        """Create a pie chart showing POS tag distribution"""
        # Count POS tags
        pos_counts = {}
        for tag in tags:
            if not tag.is_space and not tag.is_punct:
                pos_counts[tag.pos] = pos_counts.get(tag.pos, 0) + 1
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(pos_counts.keys()),
            values=list(pos_counts.values()),
            hole=0.3,
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Part-of-Speech Tag Distribution",
            font_size=12,
            showlegend=True,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_pos_bar_chart(self, tags: List[POSTag], save_path: str = None) -> go.Figure:
        """Create a bar chart showing POS tag counts"""
        # Count POS tags
        pos_counts = {}
        for tag in tags:
            if not tag.is_space and not tag.is_punct:
                pos_counts[tag.pos] = pos_counts.get(tag.pos, 0) + 1
        
        # Sort by count
        sorted_pos = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=[pos for pos, count in sorted_pos],
                y=[count for pos, count in sorted_pos],
                marker_color=px.colors.qualitative.Set3[:len(sorted_pos)]
            )
        ])
        
        fig.update_layout(
            title="Part-of-Speech Tag Counts",
            xaxis_title="POS Tag",
            yaxis_title="Count",
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_confidence_heatmap(self, tags: List[POSTag], save_path: str = None) -> go.Figure:
        """Create a heatmap showing confidence scores by POS tag"""
        # Group tags by POS and collect confidence scores
        pos_confidences = {}
        for tag in tags:
            if not tag.is_space and not tag.is_punct and tag.confidence:
                if tag.pos not in pos_confidences:
                    pos_confidences[tag.pos] = []
                pos_confidences[tag.pos].append(tag.confidence)
        
        # Calculate average confidence for each POS
        avg_confidences = {
            pos: sum(confidences) / len(confidences)
            for pos, confidences in pos_confidences.items()
        }
        
        # Create heatmap data
        pos_tags = list(avg_confidences.keys())
        confidence_values = list(avg_confidences.values())
        
        fig = go.Figure(data=go.Heatmap(
            z=[confidence_values],
            x=pos_tags,
            y=['Average Confidence'],
            colorscale='RdYlGn',
            showscale=True
        ))
        
        fig.update_layout(
            title="Average Confidence by POS Tag",
            height=200
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dependency_tree(self, tags: List[POSTag], save_path: str = None) -> go.Figure:
        """Create a dependency tree visualization"""
        # Create nodes and edges for dependency tree
        nodes = []
        edges = []
        
        for i, tag in enumerate(tags):
            if not tag.is_space:
                nodes.append({
                    'id': i,
                    'label': f"{tag.word}\n({tag.pos})",
                    'pos': tag.pos
                })
                
                # Find head (simplified - in real implementation, you'd use spaCy's dependency parser)
                if tag.dep != "ROOT":
                    # This is a simplified version - real implementation would be more complex
                    head_idx = max(0, i - 1)  # Simplified head finding
                    edges.append((head_idx, i))
        
        # Create network graph
        fig = go.Figure()
        
        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter(
                x=[nodes[edge[0]]['x'], nodes[edge[1]]['x']],
                y=[nodes[edge[0]]['y'], nodes[edge[1]]['y']],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        # Add nodes
        for node in nodes:
            fig.add_trace(go.Scatter(
                x=[node['x']],
                y=[node['y']],
                mode='markers+text',
                marker=dict(size=20, color='lightblue'),
                text=node['label'],
                textposition="middle center",
                showlegend=False
            ))
        
        fig.update_layout(
            title="Dependency Tree",
            showlegend=False,
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_word_length_analysis(self, tags: List[POSTag], save_path: str = None) -> go.Figure:
        """Create analysis of word lengths by POS tag"""
        # Group words by POS and calculate average length
        pos_lengths = {}
        for tag in tags:
            if not tag.is_space and not tag.is_punct:
                if tag.pos not in pos_lengths:
                    pos_lengths[tag.pos] = []
                pos_lengths[tag.pos].append(len(tag.word))
        
        # Calculate statistics
        pos_stats = {}
        for pos, lengths in pos_lengths.items():
            pos_stats[pos] = {
                'mean': sum(lengths) / len(lengths),
                'count': len(lengths),
                'min': min(lengths),
                'max': max(lengths)
            }
        
        # Create box plot
        fig = go.Figure()
        
        for pos, stats in pos_stats.items():
            fig.add_trace(go.Box(
                y=[stats['mean']] * stats['count'],
                name=pos,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title="Word Length Distribution by POS Tag",
            yaxis_title="Word Length",
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_comprehensive_dashboard(self, tags: List[POSTag], save_path: str = None) -> go.Figure:
        """Create a comprehensive dashboard with multiple visualizations"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('POS Distribution', 'POS Counts', 'Confidence Heatmap', 'Word Length Analysis'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "box"}]]
        )
        
        # POS Distribution (pie chart)
        pos_counts = {}
        for tag in tags:
            if not tag.is_space and not tag.is_punct:
                pos_counts[tag.pos] = pos_counts.get(tag.pos, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(pos_counts.keys()), values=list(pos_counts.values())),
            row=1, col=1
        )
        
        # POS Counts (bar chart)
        sorted_pos = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
        fig.add_trace(
            go.Bar(x=[pos for pos, count in sorted_pos], y=[count for pos, count in sorted_pos]),
            row=1, col=2
        )
        
        # Confidence Heatmap
        pos_confidences = {}
        for tag in tags:
            if not tag.is_space and not tag.is_punct and tag.confidence:
                if tag.pos not in pos_confidences:
                    pos_confidences[tag.pos] = []
                pos_confidences[tag.pos].append(tag.confidence)
        
        avg_confidences = {
            pos: sum(confidences) / len(confidences)
            for pos, confidences in pos_confidences.items()
        }
        
        fig.add_trace(
            go.Heatmap(
                z=[list(avg_confidences.values())],
                x=list(avg_confidences.keys()),
                y=['Confidence'],
                colorscale='RdYlGn'
            ),
            row=2, col=1
        )
        
        # Word Length Analysis
        pos_lengths = {}
        for tag in tags:
            if not tag.is_space and not tag.is_punct:
                if tag.pos not in pos_lengths:
                    pos_lengths[tag.pos] = []
                pos_lengths[tag.pos].append(len(tag.word))
        
        for pos, lengths in pos_lengths.items():
            fig.add_trace(
                go.Box(y=lengths, name=pos),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Comprehensive POS Analysis Dashboard",
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def export_visualization_data(self, tags: List[POSTag], output_path: str):
        """Export visualization data to JSON for external use"""
        data = {
            "pos_counts": {},
            "pos_confidences": {},
            "word_lengths": {},
            "statistics": {
                "total_words": 0,
                "unique_pos_tags": 0,
                "average_confidence": 0
            }
        }
        
        total_words = 0
        total_confidence = 0
        confidence_count = 0
        
        for tag in tags:
            if not tag.is_space and not tag.is_punct:
                # Count POS tags
                data["pos_counts"][tag.pos] = data["pos_counts"].get(tag.pos, 0) + 1
                total_words += 1
                
                # Collect confidence scores
                if tag.confidence:
                    if tag.pos not in data["pos_confidences"]:
                        data["pos_confidences"][tag.pos] = []
                    data["pos_confidences"][tag.pos].append(tag.confidence)
                    total_confidence += tag.confidence
                    confidence_count += 1
                
                # Collect word lengths
                if tag.pos not in data["word_lengths"]:
                    data["word_lengths"][tag.pos] = []
                data["word_lengths"][tag.pos].append(len(tag.word))
        
        data["statistics"]["total_words"] = total_words
        data["statistics"]["unique_pos_tags"] = len(data["pos_counts"])
        data["statistics"]["average_confidence"] = total_confidence / confidence_count if confidence_count > 0 else 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    """Demo usage of the visualizer"""
    from pos_tagger import AdvancedPOSTagger
    
    # Initialize tagger and visualizer
    tagger = AdvancedPOSTagger()
    visualizer = POSVisualizer()
    
    # Sample text
    text = "The quick brown fox jumps over the lazy dog near the riverbank."
    
    # Tag the text
    tags = tagger.tag_text(text)
    
    # Create visualizations
    print("Creating POS distribution chart...")
    fig1 = visualizer.create_pos_distribution_chart(tags, "pos_distribution.html")
    
    print("Creating POS bar chart...")
    fig2 = visualizer.create_pos_bar_chart(tags, "pos_bar_chart.html")
    
    print("Creating comprehensive dashboard...")
    fig3 = visualizer.create_comprehensive_dashboard(tags, "pos_dashboard.html")
    
    print("Exporting visualization data...")
    visualizer.export_visualization_data(tags, "visualization_data.json")
    
    print("Visualizations created successfully!")

if __name__ == "__main__":
    main()
