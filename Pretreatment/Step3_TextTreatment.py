import os
import tqdm

if __name__ == '__main__':
    load_path = 'C:\ProjectData\Pretreatment\Step2_SeparatePart'
    save_path = 'C:\ProjectData\Pretreatment\Step3_TextTreatment'
    for part_name in ['train', 'val', 'test']:
        if not os.path.exists(os.path.join(save_path, part_name)):
            os.makedirs(os.path.join(save_path, part_name, 'article'))
            os.makedirs(os.path.join(save_path, part_name, 'abstract'))
        for filename in tqdm.tqdm(os.listdir(os.path.join(load_path, part_name))):
            with open(os.path.join(load_path, part_name, filename), 'r', encoding='UTF-8') as file:
                data = file.readlines()

            if os.path.exists(os.path.join(save_path, part_name, 'article', filename)): continue
            with open(os.path.join(save_path, part_name, 'article', filename), 'w', encoding='UTF-8')as file:
                for index in range(len(data)):
                    if data[index][0:10] == '@highlight': break
                    file.write(data[index][0:-1] + ' ')

            with open(os.path.join(save_path, part_name, 'abstract', filename), 'w', encoding='UTF-8') as file:
                for index in range(len(data)):
                    if data[index][0:10] == '@highlight':
                        file.write(data[index + 2][0:-1] + '. ')
            # exit()
