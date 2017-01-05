import getopt
import sys

opts, args = getopt.getopt(sys.argv[1:], '',  ['debug', 'model='])

print sys.argv[1:]
print opts
print args
