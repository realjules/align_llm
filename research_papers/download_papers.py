import os
import requests
import time

def download_paper(url, filename):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded {filename}")
            return True
        else:
            print(f"Failed to download {filename}: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def main():
    papers = [
        # Recent Surveys
        ("https://arxiv.org/pdf/2501.09431.pdf", "responsible_llms_survey.pdf"),
        ("https://arxiv.org/pdf/2409.08087.pdf", "securing_llms.pdf"),
        
        # Speculative Decoding
        ("https://arxiv.org/pdf/2308.04623.pdf", "staged_speculative_decoding.pdf"),
        
        # Statistical Learning
        ("https://proceedings.neurips.cc/paper_files/paper/2005/file/d3d80b656929a5bc0fa34381bf42fbdd-Paper.pdf", "minimum_volume_sets.pdf"),
        
        # Attack Methods
        ("https://arxiv.org/pdf/2312.04724.pdf", "attack_methods.pdf"),
        ("https://arxiv.org/pdf/2408.01605.pdf", "attack_analysis.pdf"),
        
        # Defense Strategies
        ("https://arxiv.org/pdf/2407.13833.pdf", "defense_strategies.pdf"),
        ("https://arxiv.org/pdf/2404.13161.pdf", "defense_analysis.pdf"),
    ]
    
    for url, filename in papers:
        print(f"\nAttempting to download {filename} from {url}")
        success = download_paper(url, filename)
        if not success:
            print(f"Failed to download {filename}")
        time.sleep(2)  # Be nice to servers

if __name__ == "__main__":
    main()