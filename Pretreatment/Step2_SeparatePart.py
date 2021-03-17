import os
import tqdm
import pickle
import shutil

if __name__ == '__main__':
    load_path = 'C:\ProjectData\Pretreatment\Step0_OriginData'
    save_path = 'C:\ProjectData\Pretreatment\Step2_SeparatePart'
    if not os.path.exists(save_path):
        os.makedirs(save_path + r'\train')
        os.makedirs(save_path + r'\val')
        os.makedirs(save_path + r'\test')

    train_sha = pickle.load(open('all_train.pkl', 'rb'))
    val_sha = pickle.load(open('all_val.pkl', 'rb'))
    test_sha = pickle.load(open('all_test.pkl', 'rb'))

    # for part_name in ['dailymail']:
    for part_name in ['cnn', 'dailymail']:
        for file_name in tqdm.tqdm(os.listdir(os.path.join(load_path, part_name, 'stories'))):
            if file_name.replace('.story', '') in train_sha:
                shutil.copy(os.path.join(load_path, part_name, 'stories', file_name),
                            os.path.join(save_path, 'train', file_name))
            if file_name.replace('.story', '') in val_sha:
                shutil.copy(os.path.join(load_path, part_name, 'stories', file_name),
                            os.path.join(save_path, 'val', file_name))
            if file_name.replace('.story', '') in test_sha:
                shutil.copy(os.path.join(load_path, part_name, 'stories', file_name),
                            os.path.join(save_path, 'test', file_name))
