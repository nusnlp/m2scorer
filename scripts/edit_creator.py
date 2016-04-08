#!/usr/bin/python

import sys
import levenshtein
from getopt import getopt
from util import paragraphs
from util import smart_open
from levenshtein import equals_ignore_whitespace_casing
from levenshtein import levenshtein_matrix, edit_graph, merge_graph, transitive_arcs
import levenshtein
from itertools import izip
def print_usage():
    print >> sys.stderr, "Usage: m2scorer.py [OPTIONS] source target"
    print >> sys.stderr, "where"
    print >> sys.stderr, "  target          -   system output, sentence per line"
    print >> sys.stderr, "  source          -   source sentences with gold token edits"

    print >> sys.stderr, "OPTIONS"
    print >> sys.stderr, "  -v    --verbose                   	-  print verbose output"
    print >> sys.stderr, "        --very_verbose              	-  print lots of verbose output"
    print >> sys.stderr, "        --max_unchanged_words N     	-  Maximum unchanged words when extraction edit. Default 0."
    print >> sys.stderr, "        --beta B                    	-  Beta value for F-measure. Default 0.5."
    print >> sys.stderr, "        --ignore_whitespace_casing  	-  Ignore edits that only affect whitespace and caseing. Default no."


max_unchanged_words=0
beta = 0.5
ignore_whitespace_casing= True
verbose = False
very_verbose = False
filter_etypes = ["all"]
opts, args = getopt(sys.argv[1:], "v", ["max_unchanged_words=", "beta=", "verbose", "ignore_whitespace_casing", "very_verbose", "error_type="])
#print opts
for o, v in opts:
    if o in ('-v', '--verbose'):
        verbose = True
    elif o == '--very_verbose':
        very_verbose = True
    elif o == '--max_unchanged_words':
        max_unchanged_words = int(v)
    elif o == '--beta':
        beta = float(v)
    elif o == '--ignore_whitespace_casing':
        ignore_whitespace_casing = True
    elif o == '--error_type':
		filter_etypes = v.split(",")
    else:
        print >> sys.stderr, "Unknown option :", o
        print_usage()
        sys.exit(-1)

# starting point
if len(args) != 2:
    print_usage()
    sys.exit(-1)

source_file = args[0]
target_file = args[1]

# read the input files
system_read = open(target_file, 'r')
source_read = open(source_file, 'r')

count = 0
for candidate, source in izip(system_read, source_read):
    count += 1
    if not count % 1000:
        print >> sys.stderr, count,
    candidate = candidate.strip()
    source = source.strip()

    candidate_tok = candidate.split()
    source_tok = source.split()
    #lmatrix, backpointers = levenshtein_matrix(source_tok, candidate_tok)
    lmatrix1, backpointers1 = levenshtein_matrix(source_tok, candidate_tok, 1, 1, 1)
    lmatrix2, backpointers2 = levenshtein_matrix(source_tok, candidate_tok, 1, 1, 2)

    #V, E, dist, edits = edit_graph(lmatrix, backpointers)
    V1, E1, dist1, edits1 = edit_graph(lmatrix1, backpointers1)
    V2, E2, dist2, edits2 = edit_graph(lmatrix2, backpointers2)

    V, E, dist, edits = merge_graph(V1, V2, E1, E2, dist1, dist2, edits1, edits2)
    V, E, dist, edits = transitive_arcs(V, E, dist, edits, max_unchanged_words, very_verbose)

    print "S {0}".format(source)
    print "T {0}".format(candidate)

    # Find the shortest path with an empty gold set
    gold = []
    localdist = levenshtein.set_weights(E, dist, edits, gold, verbose, very_verbose)
    editSeq = levenshtein.best_edit_seq_bf(V, E, localdist, edits, very_verbose)
    if ignore_whitespace_casing:
        editSeq = filter(lambda x : not equals_ignore_whitespace_casing(x[2], x[3]), editSeq)

    for ed in list(reversed(editSeq)):
        # Only print those "changed" edits
        if ed[2] != ed[3]:
            print "A {0} {1}|||{2}|||{3}|||{4}".format(ed[0], ed[1], ed[2], ed[3], 0)
    print ""
system_read.close()
source_read.close()

