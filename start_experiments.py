from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--config', type=str, default='attack_best_amazon_boys_girls_amr')
args = parser.parse_args()

run_experiment(f"config_files/{args.config}.yml")
