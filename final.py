from collections import defaultdict
from stanfordcorenlp import StanfordCoreNLP
import json

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

#Run in cmd "java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators"


class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port, timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,depparse,natlog,openie',
            'pipelineLanguage': 'en',
            "openie.triple.strict": "true",
            "openie.max_entailments_per_clause": "1",
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
            return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
            return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
            return self.nlp.ner(sentence)

    def parse(self, sentence):
            return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
            return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
            return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
            tokens = defaultdict(dict)
            for token in _tokens:
                tokens[int(token['index'])] = {
                    'word': token['word'],
                    'lemma': token['lemma'],
                    'pos': token['pos'],
                    'ner': token['ner']
                }
            return tokens

###
# The text document will undergo nltk techniques
###
def pre_processing_doc1():
    # reading text
    file = open('sample.txt', 'r')
    text = file.read()

    # Annotation
    nlp_annotate(str(text), "output1.txt")

def pre_processing_doc2():
    # reading text
    file = open('sample2.txt', 'r')
    text = file.read()

    # Annotation
    nlp_annotate(str(text), "output2.txt")

def nlp_annotate(filtered_text,output_loc):
    snlp = StanfordNLP()
    output = snlp.annotate(filtered_text)
    count = len(output['sentences'])

    file1 = open(output_loc, "w")
    for x in range(count):
        result = [output['sentences'][x]['openie'] for item in output]
        for i in result:
            for rel in i:
                relationSent = rel['relation'], rel['subject'], rel['object']
                print(relationSent)
                file1.write(rel['relation']+","+rel['subject']+","+rel['object']+", Line#:"+str(x)+","+'\n')

    file1.close()

def create_graph(output_loc):

    data = pd.read_csv(output_loc, sep=',')
    data.columns = ['relation', 'Entity_1', 'Entity_2', 'line', 'id']
    data = data[['relation', 'Entity_1', 'Entity_2']]
    # print(data )
    graph = nx.from_pandas_edgelist(data, source='Entity_1', target='Entity_2', edge_attr='relation',
                                     create_using=nx.MultiGraph())
    plt.figure(figsize=(10, 9))

    pos = nx.fruchterman_reingold_layout(graph)
    nx.draw_networkx(graph, pos)
    nx.draw_networkx(graph)
    plt.savefig("C://Users//Harsimrat Kohli//Desktop//IR  Output//map_"+output_loc+".png", format="png", dpi=300)
    # plt.show()
    return graph

def select_k(spectrum, minimum_energy = 0.9):
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_total += spectrum[i]
        if running_total / total >= minimum_energy:
            return i + 1
    return len(spectrum)

def compare_graphs(graph1, graph2):
    laplacian1 = nx.spectrum.laplacian_spectrum(graph1)
    laplacian2 = nx.spectrum.laplacian_spectrum(graph2)
    k1 = select_k(laplacian1)
    k2 = select_k(laplacian2)
    k = min(k1, k2)
    similarity = sum((laplacian1[:k] - laplacian2[:k]) ** 2)
    print("Is Isomorphic: "+str(nx.could_be_isomorphic(graph1, graph2)))
    print("Similarity Metric: "+str(similarity))


if __name__ == '__main__':
    pre_processing_doc1()
    pre_processing_doc2()
    g1 = create_graph("output1.txt")
    g2 = create_graph("output2.txt")
    compare_graphs(g1, g2)
