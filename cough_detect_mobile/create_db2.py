import argparse
import json
import os.path

from tqdm import tqdm

from utils import *

# loading config file
parser = argparse.ArgumentParser()
parser.add_argument('-config',
                    type=str,
                    default='config.json',
                    help='store a json file with all the necessary parameters')
args = parser.parse_args()

# loading configuration
with open(args.config) as json_data_file:
    config = json.load(json_data_file)
config_db = config["dataset"]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))


def _string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def create_dataset(files1,
                   files0,
                   db_name,
                   hop_length=config_db["HOP"],
                   bands=config_db["BAND"],
                   window=config_db["WINDOW"],
                   nfft=config_db["NFFT"],
                   db_full_path=config["ROOT_DIR"],
                   version=config["DB_version"],
                   device_cv_name_extension = ""
                   ):
    """
     load, preprocess, normalize a sample
     input: a list of strings
     output: the processed features from each sample path in the input
    """

    print('save %s samples' % db_name)
    db_filename = os.path.join(db_full_path, db_name + version + device_cv_name_extension+'.tfrecords')
    writer = tf.python_io.TFRecordWriter(db_filename)

    def store_example(files, label):
        for f in tqdm(files):
            mfcc = preprocess(f, bands=bands, hop_length=hop_length, window=window, nfft=nfft)
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(bands),
                'width': _int64_feature(mfcc.shape[1]),
                'depth': _int64_feature(1),
                'data': _floats_feature(mfcc),
                'label': _int64_feature(label),
            }))
            writer.write(example.SerializeToString())

    store_example(files1, 1)
    store_example(files0, 0)
    writer.close()


def test_shape(files1,
               sound_id=0,  # list id of the sample that should be displayed
               hop_length=config_db["HOP"],
               bands=config_db["BAND"],
               window=config_db["WINDOW"],
               nfft=config_db["NFFT"],
               ):
    import matplotlib.pyplot as plt
    import librosa.display

    f = files1[sound_id]
    mfcc = preprocess(f, bands=bands, hop_length=hop_length, window=window, nfft=nfft)

    print('mfcc shape: ' + str(mfcc.shape))
    print('mfcc max: ' + str(np.max(mfcc)))

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


def print_stats(test_coughs, test_other, train_coughs, train_other, name='', device = ''):
    print()
    if device:
        print('----------------------------device: %s' %device )
    print('------------------------------------------------------------------')
    print('PARTITION: ' + str(name))
    print('nr of samples coughing (test): %d' % len(test_coughs))
    print('nr of samples NOT coughing (test): %d' % len(test_other))
    print('nr of samples coughing (train): %d' % len(train_coughs))
    print('nr of samples NOT coughing (train): %d' % len(train_other))
    t1 = len(test_coughs) + len(test_other)
    t2 = len(train_coughs) + len(train_other)
    print('total nr of samples: (train) %d + (test) %d = (total) %d' % (t2, t1, t1 + t2))
    print('------------------------------------------------------------------')
    print()


def main(unused_args):
    tf.set_random_seed(0)

    # Store data to TFRECORDS
    name = config_db["split_id"]
    testListCough, testListOther, trainListCough, trainListOther = get_imgs(split_id=name,
                                                                            db_root_dir=config_db["DB_ROOT_DIR"],
                                                                            listOfParticipantsInTestset=config_db[
                                                                                "test"],
                                                                            listOfParticipantsInValidationset=config_db[
                                                                                "validation"],
                                                                            listOfAllowedSources=config_db[
                                                                                "allowedSources"]
                                                                            )
    if config_db["CREATE_DB"]:
        create_dataset(trainListCough, trainListOther, 'train_' + str(name))
        create_dataset(testListCough, testListOther, 'test_' + str(name))
        print_stats(testListCough, testListOther, trainListCough, trainListOther, name)
    else:
        print_stats(testListCough, testListOther, trainListCough, trainListOther, name)
        test_shape(trainListCough)
        test_shape(trainListOther)


def main_device_cv(device):
    tf.set_random_seed(0)

    # Store data to TFRECORDS
    name = config_db["split_id"]
    testListCough, testListOther, trainListCough, trainListOther = get_imgs(split_id=name,
                                                                            db_root_dir= config_db["DB_ROOT_DIR"] ,
                                                                            listOfParticipantsInTestset=config_db[
                                                                                "test"],
                                                                            listOfParticipantsInValidationset=config_db[
                                                                                "validation"],
                                                                            listOfAllowedSources=config_db[
                                                                                "allowedSources"],
                                                                            device_cv = True,
                                                                            device = device
                                                                            )
    if config_db["CREATE_DB"]:
        create_dataset(trainListCough, trainListOther, 'train_' + str(name), device_cv_name_extension = device)
        create_dataset(testListCough, testListOther, 'test_' + str(name), device_cv_name_extension = device)
        print_stats(testListCough, testListOther, trainListCough, trainListOther, name, device = device)
    else:
        print_stats(testListCough, testListOther, trainListCough, trainListOther, name, device = device)
        test_shape(trainListCough)
        test_shape(trainListOther)



if __name__ == '__main__':

    if config["DEVICE_CV"]:
        for device in config["dataset"]["allowedSources"]:
            if device == "audio track":
                continue
            main_device_cv(device)
    else:
        tf.app.run()
