"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

import os

import tensorflow as tf
from PIL import Image

import numpy as np
import random

np.random.seed(42)
random.seed(42)


class Sampler:
    def __init__(self, indexed_ratings, item_indices, cnn_features_path, epochs):
        self._indexed_ratings = indexed_ratings
        self._item_indices = item_indices
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

        self._cnn_features_path = cnn_features_path
        self._epochs = epochs

    def read_features_triple(self, user, pos, neg):
        # load positive and negative item features
        feat_pos = np.load(self._cnn_features_path + str(pos.numpy()) + '.npy')
        feat_neg = np.load(self._cnn_features_path + str(neg.numpy()) + '.npy')

        return user.numpy(), pos.numpy(), feat_pos, neg.numpy(), feat_neg

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        actual_inter = (events // batch_size) * batch_size * self._epochs

        counter_inter = 1

        def sample():
            u = r_int(n_users)
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                sample()
            i = ui[r_int(lui)]

            j = r_int(n_items)
            while j in ui:
                j = r_int(n_items)
            return u, i, j

        for ep in range(self._epochs):
            for _ in range(events):
                yield sample()
                if counter_inter == actual_inter:
                    return
                else:
                    counter_inter += 1

    def pipeline(self, num_users, batch_size):
        def load_func(u, p, n):
            b = tf.py_function(
                self.read_features_triple,
                (u, p, n,),
                (np.int64, np.int64, np.float32, np.int64, np.float32)
            )
            return b

        data = tf.data.Dataset.from_generator(generator=self.step,
                                              output_shapes=((), (), ()),
                                              output_types=(tf.int64, tf.int64, tf.int64),
                                              args=(num_users, batch_size))
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    def step_eval(self):
        for i in self._item_indices:
            yield i
        # for i in range(50):
            # yield i

    # this is only for evaluation
    def pipeline_eval(self, batch_size):
        def load_func(i):
            b = tf.py_function(
                self.read_features,
                (i,),
                (np.int64, np.float32)
            )
            return b

        data = tf.data.Dataset.from_generator(generator=self.step_eval,
                                              output_shapes=(()),
                                              output_types=tf.int64)
        data = data.map(load_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    #pipeline_defense_eval_by_feat
    def pipeline_defense_eval_by_feat(self, denoised_feat_src_folder, output_image_size, batch_size):
        self._denoised_features_path = denoised_feat_src_folder

        def load_func_feat(i):
            b = tf.py_function(
                self.read_denoised_features,
                (i,),
                (np.int64, np.float32)
            )
            return b

        data = tf.data.Dataset.from_generator(generator=self.step_eval,
                                              output_shapes=(()),
                                              output_types=tf.int64)
        data = data.map(load_func_feat, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    def pipeline_defense_eval(self, images_src_folder, output_image_size, batch_size):
        self._images_path = images_src_folder
        self._output_image_size = output_image_size

        def load_func_img(i):
            b = tf.py_function(
                self.read_images,
                (i,),
                (np.int64, np.float32)
            )
            return b

        data = tf.data.Dataset.from_generator(generator=self.step_eval,
                                              output_shapes=(()),
                                              output_types=tf.int64)
        data = data.map(load_func_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    # this is only for evaluation
    def read_features(self, item):
        feat = np.load(self._cnn_features_path + str(item.numpy()) + '.npy')

        return item, feat

    def read_denoised_features(self, item):
        feat = np.load(self._denoised_features_path + str(item.numpy()) + '.npy')

        return item, feat

    def read_images(self, image):
        # load positive and negative item images
        im = Image.open(self._images_path + str(image.numpy()) + '.jpg')

        if im.mode != 'RGB':
            im = im.convert(mode='RGB')

        try:
            im.load()
        except ValueError:
            print(f'Image at path {image}.jpg was not loaded correctly!')

        res_sample = im.resize(self._output_image_size, resample=Image.BICUBIC)
        norm_sample = tf.keras.applications.resnet50.preprocess_input(np.array(res_sample))

        return image.numpy(), norm_sample

    def pipeline_attack(self, target_items, images_src_folder, output_image_size, batch_size):
        self._images_path = images_src_folder
        self._output_image_size = output_image_size

        def load_func_img(i):
            b = tf.py_function(
                self.read_images,
                (i,),
                (np.int64, np.float32)
            )
            return b

        data = tf.data.Dataset.from_tensor_slices(target_items)
        data = data.map(load_func_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    def pipeline_attack_no_tf(self, target_items, images_src_folder, output_image_size):
        # load positive and negative item images
        for num, target_item in enumerate(target_items):
            im = Image.open(images_src_folder + str(target_item) + '.jpg')

            if im.mode != 'RGB':
                im = im.convert(mode='RGB')

            try:
                im.load()
            except ValueError:
                print(f'Image at path {target_item}.jpg was not loaded correctly!')

            res_sample = im.resize(output_image_size, resample=Image.BICUBIC)
            norm_sample = tf.expand_dims(tf.keras.applications.resnet50.preprocess_input(np.array(res_sample)), axis=0)

            yield num, target_item, norm_sample

    def read_train_defense_images(self, image):
        # Load Original Image
        im = Image.open(self._images_path + str(image.numpy()) + '.jpg')

        if im.mode != 'RGB':
            im = im.convert(mode='RGB')

        try:
            im.load()
        except ValueError:
            print(f'Image at path {image}.jpg was not loaded correctly!')

        res_sample = im.resize(self._output_image_size, resample=Image.BICUBIC)
        norm_sample = tf.keras.applications.resnet50.preprocess_input(np.array(res_sample))

        # Load Attack Image
        attack_image_name = np.random.choice(os.listdir(os.path.join(self._attacked_images_path, str(image.numpy()))))
        # attacked_image = tf.keras.preprocessing.image.load_img(os.path.join(self._attacked_images_path, str(image.numpy()), attack_image_name))
        im = tf.keras.applications.resnet50.preprocess_input(
            np.array(Image.open(os.path.join(self._attacked_images_path, str(image.numpy()), attack_image_name))))

        return image.numpy(), norm_sample, np.array(im)

    def new_read_train_defense_images(self, image):
        # Load Original Image
        im = Image.open(self._images_path + str(image.numpy()) + '.jpg')

        if im.mode != 'RGB':
            im = im.convert(mode='RGB')

        try:
            im.load()
        except ValueError:
            print(f'Image at path {image}.jpg was not loaded correctly!')

        res_sample = im.resize(self._output_image_size, resample=Image.BICUBIC)
        norm_sample = tf.keras.applications.resnet50.preprocess_input(np.array(res_sample))

        # Load Attack Image
        attack_image_name = np.random.choice(os.listdir(os.path.join(self._attacked_images_path, str(image.numpy()))))
        # im = tf.keras.applications.resnet50.preprocess_input(
        #     np.array(Image.open(os.path.join(self._attacked_images_path, str(image.numpy()), attack_image_name))))

        return image.numpy(), norm_sample, np.load(os.path.join(self._attacked_images_path, str(image.numpy()), attack_image_name))

    def pipeline_defense(self, attacked_images_src_folder, images_src_folder, output_image_size, batch_size):
        self._attacked_images_path = attacked_images_src_folder
        self._images_path = images_src_folder
        self._output_image_size = output_image_size

        def load_func_img(i):
            b = tf.py_function(
                self.new_read_train_defense_images,
                (i,),
                (np.int64, np.float32, np.float32)
            )
            return b

        items = [item for item in os.listdir(attacked_images_src_folder)]
        num_attacks = len(os.listdir(os.path.join(attacked_images_src_folder, items[0])))
        mixed_items = np.repeat(items, num_attacks)
        np.random.shuffle(mixed_items)
        mixed_items = [int(i) for i in mixed_items]

        data = tf.data.Dataset.from_tensor_slices(mixed_items)
        data = data.map(load_func_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=batch_size)

        return data

    def read_attacked_images(self, image_path):
        original_image_id = int(image_path.numpy().decode().split('/')[0])
        # Load Original Image

        im = Image.open(self._images_path + str(original_image_id) + '.jpg')

        if im.mode != 'RGB':
            im = im.convert(mode='RGB')

        try:
            im.load()
        except ValueError:
            print(f'Image at path {original_image_id}.jpg was not loaded correctly!')

        res_sample = im.resize(self._output_image_size, resample=Image.BICUBIC)
        norm_sample = tf.keras.applications.resnet50.preprocess_input(np.array(res_sample))

        # Load Attack Image
        # im = np.array(Image.open(os.path.join(self._attacked_images_path, str(image_path.numpy().decode()))))
        # im = tf.keras.applications.resnet50.preprocess_input(np.array(tf.keras.preprocessing.image.load_img(os.path.join(self._attacked_images_path, str(image_path.numpy().decode())))))
        return image_path.numpy(), original_image_id, norm_sample, np.load(
            os.path.join(self._attacked_images_path, str(image_path.numpy().decode())))

    def save_image(self, image, path):
        im = Image.fromarray(image)
        im.save(path)

    def read_image(self, path):
        im = Image.open(path)
        return im.load()

    def pipeline_test_defense(self, attacked_images_src_folder, images_src_folder, output_image_size, batch_size):
        self._attacked_images_path = attacked_images_src_folder
        self._images_path = images_src_folder
        self._output_image_size = output_image_size

        def load_func_img(i):
            b = tf.py_function(
                self.read_attacked_images,
                (i,),
                (tf.string, np.int64, np.float32, np.float32)
            )
            return b

        items = [item for item in os.listdir(attacked_images_src_folder)]
        attacked_images = []
        for item in items:
            for name_file in os.listdir(os.path.join(attacked_images_src_folder, item)):
                attacked_images.append(os.path.join(item, name_file))

        data = tf.data.Dataset.from_tensor_slices(attacked_images)
        data = data.map(load_func_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return data

    def pipeline_test_defense_no_tf(self, attacked_images_src_folder, images_src_folder, output_image_size):

        items = [item for item in os.listdir(attacked_images_src_folder)]
        attacked_images = []
        for item in items:
            for name_file in os.listdir(os.path.join(attacked_images_src_folder, item)):
                attacked_images.append(os.path.join(item, name_file))

        for i, image_path in enumerate(attacked_images):
            original_image_id = int(image_path.split('/')[0])
            # Load Original Image

            im = Image.open(images_src_folder + str(original_image_id) + '.jpg')

            if im.mode != 'RGB':
                im = im.convert(mode='RGB')

            try:
                im.load()
            except ValueError:
                print(f'Image at path {original_image_id}.jpg was not loaded correctly!')

            res_sample = im.resize(output_image_size, resample=Image.BICUBIC)
            norm_sample = tf.keras.applications.resnet50.preprocess_input(np.array(res_sample))

            yield str(original_image_id) + "-" + image_path.split("/")[1].split('.')[
                0], original_image_id, tf.expand_dims(norm_sample, axis=0), tf.expand_dims(
                np.load(os.path.join(attacked_images_src_folder, image_path)), axis=0), i, len(attacked_images)
