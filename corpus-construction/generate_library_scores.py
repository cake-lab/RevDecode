from xml.etree import ElementTree as ET
import json
import pickle
import argparse
import os


def extract_valid_functions_with_library(signature_file):
    # Read and parse the file content
    with open(signature_file, 'r') as file:
        content = file.read()

    # Wrap content in a root tag for XML parsing and parse it
    content = f'<root>{content}</root>'
    root = ET.fromstring(content)

    # List to store valid function details
    valid_functions = []
    library_name = None

    # Find the <exe> element with a <repository>
    exe_with_repo = root.find('.//exe[repository]')
    if exe_with_repo is not None:
        library_name = exe_with_repo.find('name').text
    else:
        library_name = None

    # Iterate over each <fdesc> element
    for fdesc in root.findall('.//fdesc'):
        func_name = fdesc.get('name')
        func_addr = fdesc.get('addr')
        lshcosine_elem = fdesc.find('./lshcosine')

        # Skip if any of the required attributes are missing
        if func_name is None or func_addr is None or lshcosine_elem is None:
            continue

        # Collect LSH cosine hashes
        lshcosine_hashes = [
            hash_elem.text for hash_elem in lshcosine_elem.findall('./hash')
        ]

        # Only include if there are hashes under <lshcosine>
        if not lshcosine_hashes:
            continue

        # Append valid function details with the library name
        valid_functions.append(func_name)

    return valid_functions, library_name


def main():
    parser = argparse.ArgumentParser(description="Generate library score for libraries in the given corpus.")
    parser.add_argument("--signature_dir", help="Directory containing the signature files for the corpus.")
    parser.add_argument("--output_file", help="Output file to save the library scores.")
    args = parser.parse_args()


    """ Extract functions from corpus by looking at the signature files. """
    count_info = {}
    files = os.listdir(args.signature_dir)
    for file in files:
        signature_file_path = os.path.join(args.signature_dir, file)
        valid_functions, full_library_name = extract_valid_functions_with_library(signature_file_path)
        count_info[full_library_name] = len(valid_functions)
    
    number_of_functions_in_corpus = sum(count_info.values())

    # Sort the count_info by the number of functions in the library
    library_functions = dict(sorted(count_info.items(), key=lambda item: item[1], reverse=True))

    library_scores = {}
    for library_name, number_of_functions in library_functions.items():
        library_scores[library_name] = 1 - (number_of_functions / number_of_functions_in_corpus)

    try:
        with open(args.output_file, 'w') as file:
            json.dump(library_scores, file, indent=4)
    except Exception as e:
        print(f"Error: An error occurred while saving the library scores: {e}")


if __name__ == '__main__':
    main()