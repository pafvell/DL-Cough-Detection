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
                   device_cv_name_extension = "",
                   debug = False
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
        if label:
            countCough = 0
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
            countCough = countCough + 1
            print("CLASS COUGH %s: Amount of files %i:, Amount of samples: %i" %(db_name, len(files), countCough) )

        else:
            countOther = 0
            for f in tqdm(files):
                try:
                    samples, sample_rate = librosa.load(f, mono=True, res_type='kaiser_fast')
                except Exception as e:

                    print(f + " " + e)
                    continue

                    ###############################################################################################################

                if len(samples) <  2 * window * sample_rate:


                    indMax = np.argmax(np.abs(samples))

                    indMaxInOriginalFile = indMax

                    halfwindow_ind = int(window / 2.0 * sample_rate)
                    startind_newcut = max(indMaxInOriginalFile - halfwindow_ind, 0)
                    endind_newcut = startind_newcut + int(window * sample_rate)

                    if endind_newcut > len(samples) - 1:
                        endind_newcut = len(samples) - 1
                        startind_newcut = endind_newcut - int(window * sample_rate)

                    cutSignal = samples[startind_newcut:endind_newcut]



                    mfcc = preprocess_array(cutSignal, bands=bands, hop_length=hop_length, window=window, nfft=nfft)
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'height': _int64_feature(bands),
                        'width': _int64_feature(mfcc.shape[1]),
                        'depth': _int64_feature(1),
                        'data': _floats_feature(mfcc),
                        'label': _int64_feature(label),
                    }))
                    writer.write(example.SerializeToString())
                    countOther = countOther + 1

                ###############################################################################################################
                else:

                    iterations = range(0, len(samples), int(window * sample_rate))

                    partCounter = 0
                    for k in iterations:
                        cutSignal = samples[k: k + int(window * sample_rate)]

                        if k + int(window * sample_rate) > len(samples) - 1:
                            cutSignal = samples[-int(window * sample_rate):]

                        if debug:
                            tmpFolder = config["ROOT_DIR"] + os.sep + "tmp"
                            if not os.path.exists(tmpFolder):
                                os.makedirs(tmpFolder, 0o755)

                            librosa.output.write_wav(
                                tmpFolder + os.sep + f.split(os.sep)[-1] + "Part" + str(partCounter) + ".wav", cutSignal,
                                sample_rate)
                            partCounter = partCounter + 1

                        mfcc = preprocess_array(cutSignal, bands=bands, hop_length=hop_length, window=window, nfft=nfft, sample_rate = sample_rate)
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'height': _int64_feature(bands),
                            'width': _int64_feature(mfcc.shape[1]),
                            'depth': _int64_feature(1),
                            'data': _floats_feature(mfcc),
                            'label': _int64_feature(label),
                        }))
                        writer.write(example.SerializeToString())

                        countOther = countOther + 1

            print("CLASS OTHER %s: Amount of files %i:, Amount of samples: %i" % (db_name, len(files), countOther))


    store_example(files0, 0)
    store_example(files1, 1)

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


def main_device_cv(device, second_device=""):
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
                                                                            device = device,
                                                                            second_device = second_device
                                                                            )
    if config_db["CREATE_DB"]:
        create_dataset(trainListCough, trainListOther, 'train_' + str(name), device_cv_name_extension = device+second_device)
        if not second_device:
            create_dataset(testListCough, testListOther, 'test_' + str(name), device_cv_name_extension = device+second_device)
        print_stats(testListCough, testListOther, trainListCough, trainListOther, name, device = device+second_device)
    else:
        print_stats(testListCough, testListOther, trainListCough, trainListOther, name, device = device+second_device)
        test_shape(trainListCough)
        test_shape(trainListOther)



if __name__ == '__main__':

    if config["DEVICE_CV"]:
        for device in config["dataset"]["allowedSources"]:
            if device == "audio track":
                continue
            main_device_cv(device)
    elif config["DEVICE_CV_EXP2"]:

        for device in config["dataset"]["allowedSources"]:
            for device2 in config["dataset"]["allowedSources"]:
                if device == "audio track" or device2 == "audio track" or device == device2:
                    continue
                main_device_cv(device, second_device = device2)


    else:
        tf.app.run()
