import os
import logging
import subprocess
import argparse
import time


ghidra_project = '/Path_to_project/Sample_binary/ghidra_project'

sample_binary_set_binaries_path = '/Path_to_folder_containing_sample_binary_binaries'

libraries = os.listdir(sample_binary_set_binaries_path)

pre_command = f'analyzeHeadless {ghidra_project} postgres_object_files -process'


def main():
    parser = argparse.ArgumentParser(description='Run BSim analysis on binaries.')
    parser.add_argument('--purpose', type=str, help='The purpose of the analysis, e.g., "self_confidence" or "generate_ranking".', required=True)
    args = parser.parse_args()

    if args.purpose not in ['self_confidence', 'generate_ranking']:
        logging.error('Invalid purpose specified. Use "self_confidence" or "generate_ranking".')
        return
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'Starting BSim analysis for purpose: {args.purpose}')

    if args.purpose == 'self_confidence':
        ghidra_script = 'self_confidence.py'
    elif args.purpose == 'generate_ranking':
        ghidra_script = 'generate_ranking.py'

    for library in libraries:
        command = f'{pre_command} {library} -postScript {ghidra_script} -noanalysis'
        
        try:
            subprocess.run(command, shell=True, check=True)
            logging.info(f'Finished analyzing {library}')
        except subprocess.CalledProcessError as e:
            logging.error(f'Error while analyzing {library}: {e}')
            break
        except KeyboardInterrupt:
            logging.info('Interrupted by user, exiting...')
            break  # Break the for loop


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f'Total elapsed time: {elapsed_time:.2f} seconds')
    logging.info('All libraries have been processed successfully.')
