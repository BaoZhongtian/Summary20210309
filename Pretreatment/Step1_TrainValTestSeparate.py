import pickle
import hashlib


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf8'))
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


if __name__ == '__main__':
    for part in ['all_train.txt', 'all_val.txt', 'all_test.txt']:
        url_file = r'C:\ProjectData\cnn-dailymail-master\url_lists\%s' % part
        url_list = read_text_file(url_file)
        url_hashes = get_url_hashes(url_list)

        pickle.dump(obj=url_hashes, file=open(part.replace('txt', 'pkl'), 'wb'))
