import os
import tqdm
import pickle
import numpy

if __name__ == '__main__':
    load_path = 'C:\ProjectData\Pretreatment\Step4_BertSparateWords'
    for part in ['train', 'val', 'test']:
        total_abstract, total_article = [], []
        for filename in tqdm.tqdm(os.listdir(os.path.join(load_path, part, 'abstract'))):
            part_article = numpy.genfromtxt(
                fname=os.path.join(load_path, part, 'article', filename), dtype=int, delimiter=',')
            part_abstract = numpy.genfromtxt(
                fname=os.path.join(load_path, part, 'abstract', filename), dtype=int, delimiter=',')
            if len(part_abstract) == 0 or len(part_article) == 0: continue

            total_article.append(part_article)
            total_abstract.append(part_abstract)
        print(numpy.shape(total_article), numpy.shape(total_abstract))
        pickle.dump(total_article, open('%s-article.pkl' % part, 'wb'))
        pickle.dump(total_abstract, open('%s-abstract.pkl' % part, 'wb'))
