import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys


BATCH_SIZE = 500


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('trg_test', help='the output target embeddings')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax', 'csls'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
    parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
    parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
    parser.add_argument('-k', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
    parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    args = parser.parse_args()

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)

    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(args.seed)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not args.dot:
        embeddings.length_normalize(x)
        embeddings.length_normalize(z)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Read dictionary and compute coverage
    l1 = []
    ind = []
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    for line in f:
      if(len(line.split())==2):
        src, trg = line.split()
        l1.append(str(src)+" "+str(trg))
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)
    
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))

    # Find translations
    threshold = [0.72]
    max_transations = [1]
    for a in threshold:
      for b in max_transations:
        src = list(src2trg.keys())
        translation = collections.defaultdict(list)
        if args.retrieval == 'nn':  # Standard nearest neighbor
            for i in range(0, len(src), BATCH_SIZE):
                j = min(i + BATCH_SIZE, len(src))
                similarities = x[src[i:j]].dot(z.T)
                nn = similarities.argmax(axis=1).tolist()
                for k in range(j-i):
                    translation[src[i+k]] = nn[k]
        elif args.retrieval == 'invnn':  # Inverted nearest neighbor
            best_rank = np.full(len(src), x.shape[0], dtype=int)
            best_sim = np.full(len(src), -100, dtype=dtype)
            for i in range(0, z.shape[0], BATCH_SIZE):
                j = min(i + BATCH_SIZE, z.shape[0])
                similarities = z[i:j].dot(x.T)
                ind = (-similarities).argsort(axis=1)
                ranks = asnumpy(ind.argsort(axis=1)[:, src])
                sims = asnumpy(similarities[:, src])
                for k in range(i, j):
                    for l in range(len(src)):
                        rank = ranks[k-i, l]
                        sim = sims[k-i, l]
                        if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
                            best_rank[l] = rank
                            best_sim[l] = sim
                            translation[src[l]] = k
        elif args.retrieval == 'invsoftmax':  # Inverted softmax
            sample = xp.arange(x.shape[0]) if args.inv_sample is None else xp.random.randint(0, x.shape[0], args.inv_sample)
            partition = xp.zeros(z.shape[0])
            for i in range(0, len(sample), BATCH_SIZE):
                j = min(i + BATCH_SIZE, len(sample))
                partition += xp.exp(args.inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
            for i in range(0, len(src), BATCH_SIZE):
                j = min(i + BATCH_SIZE, len(src))
                p = xp.exp(args.inv_temperature*x[src[i:j]].dot(z.T)) / partition
                nn = p.argmax(axis=1).tolist()
                for k in range(j-i):
                    translation[src[i+k]] = nn[k]
        elif args.retrieval == 'csls':  # Cross-domain similarity local scaling
            knn_sim_bwd = xp.zeros(z.shape[0])
            for i in range(0, z.shape[0], BATCH_SIZE):
                j = min(i + BATCH_SIZE, z.shape[0])
                knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
            for i in range(0, len(src), BATCH_SIZE):
                j = min(i + BATCH_SIZE, len(src))
                similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
                nn = similarities.argmax(axis=1).tolist()
                for k in range(j-i):
                  
                  ind = np.argsort(-similarities[k])
                  # print(nn[k] , type(nn[k]))
                  l = []             
                  cnt = 0
                  for u in ind : 
                    u = int(u)
                    cnt = cnt+1
                    if(similarities[k][u] < a) : 
                      break
                    l.append(u)
                    if(cnt == b):
                      break
                    
                  if(len(l) == 0):
                    l.append(nn[k])

                  translation[src[i+k]] = l
                  # translation[src[i+k]] = nn[k]
        
        l2 = []
        l3 = []
        trgfile = open(args.trg_test, mode='w')
        for i , j in translation.items():
            for k in j :
              l3.append(str(src_words[i]))
              # if(k == -1):
              #   l2.append(str(src_words[i]) + " " + str(src_words[i]))
              #   trgfile.write(str(src_words[i]) + " " + str(src_words[i]))
              #   trgfile.write("\n")
              #   continue
              l2.append(str(src_words[i]) + " " + str(trg_words[k]))
              trgfile.write(str(src_words[i]) + " " + str(trg_words[k]))
              trgfile.write("\n")

        # for i in l1 :
        #   src , trg = i.strip().split()
        #   if (src not in l3 ) :
        #     trgfile.write(str(src) +" " + str(src))
        #     trgfile.write("\n")
        #     l3.append(src)        
        trgfile.close()

        file_1 = open(args.trg_test, mode='r')
        file_2 = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')

        l1 = []
        l2 = []
        for line in file_1 :
          src,trg = line.split()
          l1.append(src+" "+trg)
        for line in file_2 :
          src,trg = line.split()
          l2.append(src+" "+trg)
        cnt = 0
        for i in l1 :
          if i in l2 :
            cnt = cnt+1
        ln = len(l1)
        precision = cnt/ln
        print(cnt)
        print(ln)
        print("Threshold " , a , " Max_Translations ",b )
        print(' Precision:{0:.2f}%'.format(precision*100))
        ln1 = len(l2)
        recall = cnt/ln1
        print(cnt)
        print(ln1)
        print(' Recall:{0:.2f}%'.format(recall*100))
        print(' F1_Score:',format((2*precision*recall)/(precision+recall)))

if __name__ == '__main__':
    main()
