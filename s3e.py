
import numpy as np
import argparse

from utils import load_file, create_dictionary, load_wordvec, load_word_weight
from utils import semantic_construction, compute_embedding


PATH_TO_VEC = [#'./word_embedding/glove.840B.300d.txt', # GloVe Vector
                './word_embedding/crawl-300d-2M.vec', # FastText Vector
                './word_embedding/lexvec.commoncrawl.300d.W.pos.vectors', # LexVec Vector
                './word_embedding/paragram_300_sl999.txt', # PSL Vector
                ]
PATH_TO_WORD_WEIGHTS = './word_embedding/enwiki_vocab_min200.txt' # Word Weights Vector
PATH_TO_SENTENCE = './custrev.pos'

if __name__ == "__main__":
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_num", default=10, type=int,
                        help="number of semantic groups to construct")
    parser.add_argument("--postprocessing", default=1, type=int,
                        help="principal component removal")
    args = parser.parse_args()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Load text file
    sentences = load_file(PATH_TO_SENTENCE)

    # Load dictionary
    args.id2word, args.word2id = create_dictionary(sentences)

    # Load word vectors
    args.word_vec_np = load_wordvec(PATH_TO_VEC, args.word2id)
    args.wvec_dim = args.word_vec_np.shape[1]

    # Load word weights
    args.word_weight = load_word_weight(PATH_TO_WORD_WEIGHTS, args.word2id, a=1e-3)
    
    # Construct semantic groups
    semantic_construction(args)

    # Generate embedding
    sentence_emb = compute_embedding(args, sentences)
    


    # Provide Example
    index1 = int(input("\nThe index for the first sentence:"))
    print("The first sentence is:", " ".join(sentences[index1]))
    index2 = int(input("The index for the second sentence:"))
    print("The second sentence is:", " ".join(sentences[index2]))

    print("The similarity between them are:", sentence_emb[index1].dot(sentence_emb[index2])/np.linalg.norm(sentence_emb[index1])/np.linalg.norm(sentence_emb[index2]))


