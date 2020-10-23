#!/usr/bin/env python

from __future__ import print_function

import argparse

from levenshtein import batch_multi_pre_rec_f1


def smart_open(fname, mode='rb'):
    if fname.endswith('.gz'):
        import gzip
        return gzip.open(fname, mode, 1)
    else:
        return open(fname, mode)


def paragraphs(lines, is_separator=lambda x : x == '\n', joiner=''.join):
    paragraph = []
    for line in lines:
        if is_separator(line):
            if paragraph:
                yield joiner(paragraph)
                paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        yield joiner(paragraph)


def load_annotation(gold_file):
    source_sentences = []
    gold_edits = []
    fgold = smart_open(gold_file)
    puffer = fgold.read()
    fgold.close()
    puffer = puffer.decode('utf8')
    for item in paragraphs(puffer.splitlines(True)):
        item = item.splitlines(False)
        sentence = [line[2:].strip() for line in item if line.startswith('S ')]
        assert sentence != []
        annotations = {}
        for line in item[1:]:
            if line.startswith('I ') or line.startswith('S '):
                continue
            assert line.startswith('A ')
            line = line[2:]
            fields = line.split('|||')
            start_offset = int(fields[0].split()[0])
            end_offset = int(fields[0].split()[1])
            etype = fields[1]
            if etype == 'noop':
                start_offset = -1
                end_offset = -1
            corrections = [c.strip() if c != '-NONE-' else '' for c in fields[2].split('||')]
            # NOTE: start and end are *token* offsets
            original = ' '.join(' '.join(sentence).split()[start_offset:end_offset])
            annotator = int(fields[5])
            if annotator not in list(annotations.keys()):
                annotations[annotator] = []
            annotations[annotator].append((start_offset, end_offset, original, corrections))
        tok_offset = 0
        for this_sentence in sentence:
            tok_offset += len(this_sentence.split())
            source_sentences.append(this_sentence)
            this_edits = {}
            for annotator, annotation in annotations.items():
                this_edits[annotator] = [edit for edit in annotation
                                         if 0 <= edit[0] <= tok_offset and 0 <= edit[1] <= tok_offset]
            if len(this_edits) == 0:
                this_edits[0] = []
            gold_edits.append(this_edits)
    return source_sentences, gold_edits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('system')
    parser.add_argument('gold_m2')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--ignore-whitespace-casing', action='store_true')
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--timeout', type=float, default=5)
    parser.add_argument('--max-unchanged-words', type=int, default=2)

    args = parser.parse_args()

    # load source sentences and gold edits
    source_sentences, gold_edits = load_annotation(args.gold_m2)

    # load system hypotheses
    fin = smart_open(args.system)
    system_sentences = [line.decode("utf8").strip() for line in fin.readlines()]
    fin.close()

    p, r, f1 = batch_multi_pre_rec_f1(system_sentences, source_sentences, gold_edits, args.max_unchanged_words,
                                      args.beta, args.ignore_whitespace_casing, args.verbose, args.timeout)

    print("Precision   : %.4f" % p)
    print("Recall      : %.4f" % r)
    print("F_%.1f       : %.4f" % (args.beta, f1))

