import os
import sys
from process_docs import main as process_docs_main
from rbf_network import main as rbf_network_main
from rbf_network_2d import main as rbf_network_2d_main

def setup_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['docs', 'results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    # Setup directories
    setup_directories()
    
    print("\n=== Анализ документов ===")
    process_docs_main()
    
    print("\n=== Реализация RBF сети (1D) ===")
    rbf_network_main()
    
    print("\n=== Реализация RBF сети (2D) - Лабораторная работа №2 ===")
    rbf_network_2d_main()
    
    # Move generated plots to results directory
    try:
        for file in os.listdir('.'):
            if file.endswith('.png'):
                os.rename(file, os.path.join('results', file))
        print("\nРезультаты сохранены в директории 'results'")
    except Exception as e:
        print(f"Ошибка при перемещении файлов: {e}")

if __name__ == "__main__":
    main()
