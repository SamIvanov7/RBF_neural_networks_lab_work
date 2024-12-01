from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

def read_pdf(file_path):
    """Read and extract text from a PDF document."""
    try:
        print(f"\nAttempting to read {file_path}...")
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        error_msg = f"Error reading {os.path.basename(file_path)}: {str(e)}"
        print(f"Warning: {error_msg}")
        return error_msg

def analyze_text(text):
    """Perform basic text analysis."""
    # Word count
    words = text.split()
    word_count = len(words)

    # Character count
    char_count = len(text)

    # Word frequency
    word_freq = Counter(words)
    most_common = word_freq.most_common(10)

    return {
        'word_count': word_count,
        'char_count': char_count,
        'most_common_words': most_common
    }

def create_word_freq_plot(filename, most_common_words):
    """Create word frequency plot."""
    if most_common_words:
        words, counts = zip(*most_common_words)
        plt.figure(figsize=(12, 6))
        plt.bar(words, counts)
        plt.title(f'Most Common Words in {filename}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_file = f'{os.path.splitext(filename)[0]}_word_freq.png'
        plt.savefig(output_file)
        plt.close()
        print(f"Word frequency plot saved as: {output_file}")

def main():
    # File paths
    files = [
        './docs/Vvedenie.pdf',
        './docs/Lab_rabota.pdf'
    ]

    # Process documents
    for file_path in files:
        filename = os.path.basename(file_path)
        content = read_pdf(file_path)
        analysis = analyze_text(content)

        print(f"\nAnalysis for {filename}:")
        print(f"Word count: {analysis['word_count']}")
        print(f"Character count: {analysis['char_count']}")
        print("\nMost common words:")
        for word, count in analysis['most_common_words']:
            print(f"{word}: {count} times")

        # Create visualization
        create_word_freq_plot(filename, analysis['most_common_words'])

if __name__ == "__main__":
    main()
