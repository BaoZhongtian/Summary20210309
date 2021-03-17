import os
import tqdm
import pytorch_pretrained_bert

if __name__ == '__main__':
    load_path = 'C:\ProjectData\Pretreatment\Step3_TextTreatment'
    save_path = 'C:\ProjectData\Pretreatment\Step4_BertSparateWords'

    tokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained('bert-large-uncased')

    for train_part in ['train', 'val', 'test']:
        for use_part in ['abstract', 'article']:
            if not os.path.exists(os.path.join(save_path, train_part, use_part)):
                os.makedirs(os.path.join(save_path, train_part, use_part))

            for filename in tqdm.tqdm(os.listdir(os.path.join(load_path, train_part, use_part))):
                if os.path.exists(
                    os.path.join(save_path, train_part, use_part, filename.replace('story', 'txt'))): continue

                with open(os.path.join(load_path, train_part, use_part, filename), 'r', encoding='UTF-8') as file:
                    data = file.readlines()
                assert len(data) == 1

                tokenized_text = tokenizer.tokenize(data[0])
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

                with open(os.path.join(save_path, train_part, use_part, filename.replace('story', 'txt')), 'w') as file:
                    for index in range(len(indexed_tokens)):
                        if index != 0: file.write(',')
                        file.write(str(indexed_tokens[index]))
                # exit()
