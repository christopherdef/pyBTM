#!/usr/bin/env python
# coding=utf-8
# translate word into id in documents
import sys
import os

w2id = {}

def indexFile(pt, res_pt, csvCol=2):
    print('index file: '+str(pt))
    i = 0
    wf = open(res_pt, 'w', encoding='utf8')
    for l in open(pt, encoding='utf8'):
        i+=1
        if i % 10_000 == 0:
            print(i, end='\r')

        ws = l.strip().split()
        for w in ws:
            if w not in w2id:
                w2id[w] = len(w2id)

        wids = [w2id[w] for w in ws]
        print(' '.join(map(str, wids)), file=wf)

    print('index write to: '+str(res_pt))

def write_w2id(res_pt):
    print('vocab write to: '+str(res_pt))
    wf = open(res_pt, 'w', encoding='utf8')
    for w, wid in sorted(w2id.items(), key=lambda d:d[1]):
        print('%d\t%s' % (wid, w), file=wf)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python %s <doc_pt> <dwid_pt> <voca_pt>' % sys.argv[0])
        print('\tdoc_pt    input docs to be indexed, each line is a doc with the format "word word ..."')
        print('\tdwid_pt   output docs after indexing, each line is a doc with the format "wordId wordId..."')
        print('\tvoca_pt   output vocabulary file, each line is a word with the format "wordId    word"')
        exit(1)

    doc_pt = sys.argv[1]
    dwid_pt = sys.argv[2]
    voca_pt = sys.argv[3]
    fmt = sys.argv[4] if len(sys.argv) > 4 else None
    csv_col = sys.argv[5] if len(sys.argv) > 5 else 2

    print('BEGINNING INDEX')
    indexFile(doc_pt, dwid_pt, csv_col)
    print('n(w)='+str(len(w2id)))
    if not os.path.exists(voca_pt):
        write_w2id(voca_pt)
