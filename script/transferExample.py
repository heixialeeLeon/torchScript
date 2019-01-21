import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
from argparse import ArgumentParser
from crnn import *  # include the model file

parser = argparse.ArgumentParser(description='TorchScript Transfer parameter')
parser.add_argument('--input_batch', type=int, default=1, help="The input batch size")
parser.add_argument('--input_channel', type=int, default=3, help="The input channel size")
parser.add_argument('--input_height', type=int, default=0, help="The input height")
parser.add_argument('--input_width', type=int, default=0, help="The input width")
parser.add_argument('--input_model', type=str, default="leon.pt",help="The input pytorch model")
parser.add_argument('--output_file', type=str, default="torchScript.pt", help="The ouput torch script file")
args = parser.parse_args()
print(args)

def ToTorchScript(model, outputFile):
    input = torch.randn(args.input_batch, args.input_channel, args.input_height, args.input_width)
    traced_script_module = torch.jit.trace(model.to(torch.device('cuda')), input.to(torch.device('cuda')))
    traced_script_module.save(outputFile)

net = torch.load(args.input_model)
net.eval()
ToTorchScript(net, args.output_file)