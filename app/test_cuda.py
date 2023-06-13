import torch
import argparse

parser = argparse.ArgumentParser(prog='Test CUDA',
                    description='Test if CUDA is working properly',
                    epilog='Jose Angel Perez Garrido - 2023')

args = parser.parse_args()

print("CUDA AVAILABILITY:",torch.cuda.is_available())