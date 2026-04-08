import argparse
from ml_model import run_experiment

def main():
    parser = argparse.ArgumentParser(description='NGBoost Regression Analysis')
    parser.add_argument('--dataset_dir', type=str, default='./dataset',
                        help='Directory containing training and test datasets')
    parser.add_argument('--dataset_type', type=str, choices=['full', 'half','full_sample143'], default='full_sample143',
                        help='Dataset type: "full" or "half"',)
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='train',
                        help='Mode to run: "train" to train and save models, "test" to evaluate saved models, '
                             '"both" to run training then testing')
    args = parser.parse_args()
    run_experiment(args)

if __name__ == '__main__':
    main()
# usage examples:    
# python main.py --dataset_dir ./dataset --dataset_type full --mode train
# python main.py --dataset_dir ./dataset --dataset_type full --mode test
# python main.py --dataset_dir ./dataset --dataset_type full --mode both

