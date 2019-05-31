"""
    Usage: 
    python data_converter.py --input_path <Where your bair dataset is> 
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from google.protobuf.json_format import MessageToDict
import pickle, os, re, time, argparse


def unpack_float_list(feature):
    return feature.float_list.value

def unpack_bytes_list(feature):
    return feature.bytes_list.value

def unpack_int64_list(feature):
    return feature.int64_list.value

def get_unpack_functions(structures):
    functions = []
    for structure in structures:
        if structure == "floatList":
            functions.append(unpack_float_list)
        elif structure == "bytesList":
            functions.append(unpack_bytes_list)
        elif structure == 'Int64List':
            functions.append(unpack_int64_list)
        else:
            raise Exception('no such type {}'.format(structure))
    return functions

class Converter(object):
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.datasets = self.get_sets()
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.sess = tf.Session()


    def get_sets(self):
        """
            Return the data folder inside the input path that we need to convert
        """
        return ('train', 'test', 'val')
    
    def check_config(self):
        """
            Check the configure in the TFrecode
        """
        # assume each file is the same
        folder = os.path.join(self.input_path, self.datasets[0])
        filename = os.path.join(folder, os.listdir(folder)[0])

        # find one sequence
        record = tf.python_io.tf_record_iterator(filename)
        traj = next(record)
        dict_message = MessageToDict(tf.train.Example.FromString(traj))
        feature = dict_message['features']['feature']
        
        # find the keys and frame numbers in the sequence
        num_set = set()
        key_set = set()
        for key in feature.keys():
            parse = re.findall(r'(\d+)(.*)', key)[0]
            num_set.add(int(parse[0]))
            key_set.add(parse[1])
        self.sequence_size = max(num_set) + 1
        self.keys = list(key_set)

        # find the data structure for each key
        self.structure = list()
        for key in self.keys:
            data = feature['0' + key]
            self.structure.append(list(data.keys())[0])
        self.functions = get_unpack_functions(self.structure)

        print('----------------------------------------------')
        print('Sequence size: {}'.format(self.sequence_size))
        for i in range(len(self.keys)):
            print(self.keys[i], self.structure[i])
        print('----------------------------------------------')

        # get image size
        for i in range(len(self.keys)):
            if 'image' in self.keys[i]:
                image_key = self.keys[i]
                image_function = self.functions[i]

        example = tf.train.Example()
        example.ParseFromString(traj)
        feature = example.features.feature
        image_raw = image_function(feature['16' + image_key])[0]
        image_flatten = self.sess.run(tf.decode_raw(image_raw, tf.uint8))

        image_size = int(np.sqrt(image_flatten.shape[0] // 3)) # assume images are square
        self.image_shape = (image_size, image_size, 3)
        image = image_flatten.reshape(self.image_shape)
        plt.imshow(image)
        plt.title('demo image')
        plt.show()

        # save the config 
        config = {}
        for key, function in  zip(self.keys, self.functions):
            raw = function(feature['0' + key])
            if 'image' in key:
                image_flatten = self.sess.run(tf.decode_raw(raw[0], tf.uint8))
                data = image_flatten.reshape(self.image_shape)
            else:
                data = np.array(raw)
            config[key] = data.shape
        print('-' * 50)
        print(config)
        print('-' * 50)
        with open(os.path.join(self.output_path, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
    def parser(self, traj):
        example = tf.train.Example()
        example.ParseFromString(traj)
        feature = example.features.feature
        dict_data = dict()
        for key, func in zip(self.keys, self.functions):
            list_data = []
            for i in range(self.sequence_size):
                raw = func(feature[('%d' % i) + key])
                if 'image' in key:
                    image_flatten = np.array([b for b in raw[0]], dtype=np.uint8)
                    data = image_flatten.reshape(self.image_shape)
                else:
                    data = np.array(raw)
                list_data.append(data[np.newaxis])
            dict_data[key] = np.concatenate(list_data, axis=0)
        return dict_data

    def convert(self):
        self.check_config()
        for dataset in self.datasets:
            print('-' * 50)
            print('In dataset {}'.format(dataset))
            print('-' * 50)
            input_folder = os.path.join(self.input_path, dataset)
            output_folder = os.path.join(self.output_path, dataset)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            filenames = sorted(os.listdir(input_folder))
            count = 0
            for filename in filenames:
                record = tf.python_io.tf_record_iterator(os.path.join(input_folder, filename))
                for traj in record:
                    start_time = time.time()
                    dict_data = self.parser(traj)
                    with open(os.path.join(output_folder, '%05d_traj.pkl' % count), 'wb') as f:
                        pickle.dump(dict_data, f)
                    count += 1 
                    print('%d sample using time: %f ms' % (count, time.time() - start_time))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str, default='data/bair/')
    args = parser.parse_args()

    converter = Converter(args.input_path, args.output_path)
    converter.convert()

if __name__ == "__main__":
    main()