import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--var1', dest='var1', action='store_true')
parser.add_argument('--var2', dest='var2', action='store_true')
arguments = parser.parse_args()
print(arguments)
