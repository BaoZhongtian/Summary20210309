import os
import tqdm
import pickle
import numpy

if __name__ == '__main__':
    load_path = 'C:\ProjectData\Pretreatment\Step4_BertSparateWords'

    dictionary = set()
    for train_part in ['train', 'val', 'test']:
        for use_part in ['abstract', 'article']:
            for filename in tqdm.tqdm(os.listdir(os.path.join(load_path, train_part, use_part))):
                data = numpy.genfromtxt(fname=os.path.join(load_path, train_part, use_part, filename), dtype=int,
                                        delimiter=',')
                for sample in data:
                    dictionary.add(sample)
        print(len(dictionary))
    pickle.dump(dictionary, open('Step5_Dictionary.pkl', 'wb'))
