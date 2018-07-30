from bs4 import BeautifulSoup
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import math


def cue_words():
    cue = {'accuraci', 'describ', 'illustr', 'origin', 'run', 'achiev', 'detail', 'improv', 'outperform', 'scenario',
           'actual', 'determin', 'increas', 'output', 'schema', 'addition', 'differ', 'infer', 'paramet', 'scheme',
           'aggreg', 'discuss', 'inform', 'partition', 'score', 'algorithm', 'distanc', 'input', 'percentag', 'slope',
           'analysi', 'distribut', 'instanc', 'perform', 'show', 'approxim', 'docum', 'interest', 'plot', 'shown',
           'assign', 'error', 'label', 'point', 'signific', 'averag', 'estim', 'larg', 'position', 'significantli',
           'baselin', 'evalu', 'larger', 'precision', 'similar', 'case', 'execut', 'length', 'predic', 'size',
           'collect', 'exist', 'level', 'previou', 'small', 'column', 'expect', 'line', 'problem', 'state', 'compar',
           'experiment', 'list', 'procedur', 'step', 'comparison', 'fact', 'maximum', 'process', 'structur', 'comput',
           'featur', 'mean', 'produc', 'system', 'concept', 'figur', 'measur', 'rang', 'tabl', 'consist', 'final',
           'method', 'rank', 'techniqu', 'constraint', 'focu', 'metric', 'rate', 'test', 'content', 'frequenc',
           'minimum', 'row', 'threshold', 'correl', 'frequent', 'model', 'record', 'time', 'cost', 'good', 'note',
           'relat', 'total', 'curv', 'graph', 'number', 'repres', 'valu', 'data', 'hierarchi', 'observ', 'requir',
           'vari', 'dataset', 'high', 'obtain', 'result', 'variabl', 'defin', 'higher', 'oper', 'return', 'x-axis',
           'depict', 'highlight', 'optim', 'rule', 'y-axis'}
    return cue


# first, remove redundant information,
# including journal meta info, article meta info, reference article info, author's contributions, supplementary material
# second, extract figure caption, reference sentences, full text
def process(xml_file):
    soup = BeautifulSoup(open(xml_file, 'r', encoding='utf8'), 'lxml')
    # article_title = soup.find('article-title').get_text()
    captions = []   # caption list
    ref_para = {}  # paragraph that reference sentence belongs to
    ps = []      # paragraph list

    # remove journal meta info, article meta info
    soup.find('front').extract()
    # remove table data
    for table_tag in soup.find_all('table-wrap'):
        if table_tag:
            table_tag.extract()
    # remove reference article info, author's contribution
    soup.find('back').extract()
    # remove supplementary-material info
    for supp_tag in soup.find_all('supplementary-material'):
        if supp_tag:
            supp_tag.extract()

    for fig_tag in soup.find_all('fig'):
        if fig_tag:
            captions.append(fig_tag.extract().get_text())

    for xref_tag in soup.find_all('xref'):
        if xref_tag['ref-type'] == 'fig':
            fig_num = xref_tag['rid']
            ref_para[fig_num] = []
    for xref_tag in soup.find_all('xref'):
        if xref_tag['ref-type'] == 'fig':
            fig_num = xref_tag['rid']
            ref_para[fig_num].append(xref_tag.find_parent('p').get_text())

    for p_tag in soup.find_all('p'):
        ps.append(p_tag.get_text())

    full_text = '\n'.join(p for p in ps)   # full text exclude caption, table data, section title and other info

    return captions, ref_para, ps, full_text


# input caption
# output query generated from caption
# stopwords removal, stemming using Porter's algorithm
def query_generation(caption):
    caption = caption.replace('Fig.', 'Fig').replace('FIG.', 'FIG')
    porter = PorterStemmer()
    stop = stopwords.words('english')
    caption = nltk.word_tokenize(caption)
    query = [porter.stem(w) for w in caption if w not in stop]
    return query


# input full text
# output sentences segmentation, words splitting
def full_text_process(text):
    text = text.replace('Fig.', 'Fig').replace('FIG.', 'FIG')
    porter = PorterStemmer()
    stop = stopwords.words('english')
    sentences = nltk.sent_tokenize(text)
    words = []
    for s in sentences:
        words.append([porter.stem(w.lower()) for w in nltk.word_tokenize(s) if w not in stop])
    return sentences, words


def bm25(query, sentences, words):
    k1 = 2
    k3 = 2
    b = 0.75
    scores = {}
    sf_t = {}       # sentence frequency
    # similarity20 = []
    similarity20 = set()

    total_sens = len(sentences)
    total_words = sum(len(s) for s in words)
    avg_sen_len = total_words / total_sens      # average length of sentences
    tf_tq = Counter(query)

    for word in sorted(tf_tq):
        sf_t[word] = 0
        for sentence in words:
            if word in sentence:
                sf_t[word] += 1

    for i, sentence in enumerate(words):
        score = 0
        # ss = sentences[i]
        for word in sorted(tf_tq):
            if word not in sentence:
                continue
            tf_ts = Counter(sentence)  # the frequency of term t in sentence s
            isf = math.log2(total_sens / sf_t[word])  # inverse sentence frequency
            fts = (k1 + 1) * tf_ts[word] / (k1 * ((1 - b) + b * (len(sentence) / avg_sen_len)) + tf_ts[word])
            ftq = (k3 + 1) * tf_tq[word] / (k3 + tf_tq[word])
            score += isf * fts * ftq
        # scores[ss] = score
        scores[i] = score

    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    scores = scores[:20]
    print(scores)
    for sentence, score in scores:
        # similarity20.append(sentence)
        similarity20.add(sentence)

    return similarity20


# sentences that include cue words or phrases
def cue_sentence(sentences, words):
    # sents = []
    sents = set()
    for i, sentence in enumerate(words):
        if cue_words().intersection(set(sentence)):
            sents.add(i)
            # sents.append(sentences[i])
    return sents


def ref_sentences(sentences):
    fig_type = {'FIGURE', 'Figure', 'FIG', 'Fig'}
    # ref_sens = []
    ref_sens = set()
    # pro_sens = []       # the first ten sentences on either side of a reference sentence
    pro_sens = set()
    for i, sentence in enumerate(sentences):
        s = nltk.word_tokenize(sentence)
        if fig_type.intersection(set(s)):
            # ref_sens.append(sentence)
            ref_sens.add(i)
            if i - 10 >= 0:
                # pro_sens.extend(sentences[i - 10: i])
                pro_sens.update(range(i - 10, i))
            else:
                # pro_sens.extend(sentences[0: i])
                pro_sens.update(range(0, i))

            # pro_sens.extend(sentences[i + 1: i + 11])
            pro_sens.update(range(i + 1, i + 11))

    return ref_sens, pro_sens


if __name__ == '__main__':
    path = r'D:\test.xml'
    # path = r'D:\test2.xml'
    caps, ref_pa, paras, ftext = process(path)
    que = query_generation(caps[1])
    # que = query_generation(caps[3])
    sens, sen_words = full_text_process(ftext)
    print(sens)
    # feature1 = bm25(que, sens, sen_words)
    # feature3 = cue_sentence(sens, sen_words)
    # feature4, feature6 = ref_sentences(sens)
    for para in ref_pa['Fig1']:
        para = para.replace('Fig.', 'Fig').replace('FIG.', 'Fig')
        for s in nltk.sent_tokenize(para):
            print(sens.index(s))
