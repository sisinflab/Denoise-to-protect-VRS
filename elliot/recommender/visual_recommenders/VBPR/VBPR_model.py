"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import tensorflow as tf
import numpy as np
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from tensorflow import keras
import random
from elliot.denoiser import UNet
from PIL import Image
from sklearn import linear_model, svm
import torch
import gc

random.seed(42)


class VBPR_model(keras.Model):
    def __init__(self, factors=200, factors_d=20,
                 learning_rate=0.001,
                 l_w=0, l_b=0, l_e=0,
                 num_image_feature=2048,
                 num_users=100,
                 num_items=100,
                 defense_learning_rate=0.001, defenses=['none'],
                 name="VBPRMF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(42)

        self._factors = factors
        self._factors_d = factors_d
        self._learning_rate = learning_rate
        self._defense_learning_rate = defense_learning_rate
        self.l_w = l_w
        self.l_b = l_b
        self.l_e = l_e
        self.num_image_feature = num_image_feature
        self._num_items = num_items
        self._num_users = num_users

        self.initializer = tf.initializers.GlorotUniform()
        self.Bi = tf.Variable(tf.zeros(self._num_items), name='Bi', dtype=tf.float32)
        self.Gu = tf.Variable(self.initializer(shape=[self._num_users, self._factors]), name='Gu', dtype=tf.float32)
        self.Gi = tf.Variable(self.initializer(shape=[self._num_items, self._factors]), name='Gi', dtype=tf.float32)

        self.Bp = tf.Variable(
            self.initializer(shape=[self.num_image_feature, 1]), name='Bp', dtype=tf.float32)
        self.Tu = tf.Variable(
            self.initializer(shape=[self._num_users, self._factors_d]),
            name='Tu', dtype=tf.float32)
        self.E = tf.Variable(
            self.initializer(shape=[self.num_image_feature, self._factors_d]),
            name='E', dtype=tf.float32)

        self.optimizer = tf.optimizers.Adam(self._learning_rate)
        self.defense_optimizer = tf.optimizers.Adam(self._defense_learning_rate)

        self._defenses = defenses
        self._ife = tf.keras.applications.ResNet50()


        # For Defense
        if 'madry' in list(self._defenses):
            from cnn_models.model import Model
            import torchvision.models as models
            model_path = 'cnn_models/madry/imagenet_linf_8.pt'
            self._madry_feature_extractor = Model(model=models.resnet50(), gpu=-1, model_path=model_path,
                                                  pretrained_name='madry')
            self._madry_feature_extractor.set_out_layer(drop_layers=1)

        if 'denoiser' in list(self._defenses):
            self._unet = UNet()

        self.feature_extractor = tf.keras.applications.ResNet50()
        self.feature_extractor = tf.keras.Model(self.feature_extractor.input,
                                                self.feature_extractor.get_layer('avg_pool').output)

    # @tf.function
    def call(self, inputs, training=None):
        user, item, feature_i = inputs
        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_u = tf.squeeze(tf.nn.embedding_lookup(self.Gu, user))
        theta_u = tf.squeeze(tf.nn.embedding_lookup(self.Tu, user))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))

        xui = beta_i + tf.reduce_sum((gamma_u * gamma_i), axis=1) + \
              tf.reduce_sum((theta_u * tf.matmul(feature_i, self.E)), axis=1) + \
              tf.squeeze(tf.matmul(feature_i, self.Bp))

        return xui, gamma_u, gamma_i, feature_i, theta_u, beta_i

    # @tf.function
    def train_step(self, batch):
        user, pos, feature_pos, neg, feature_neg = batch
        with tf.GradientTape() as t:
            xu_pos, gamma_u, gamma_pos, _, theta_u, beta_pos = \
                self(inputs=(user, pos, feature_pos), training=True)
            xu_neg, _, gamma_neg, _, _, beta_neg = self(inputs=(user, neg, feature_neg), training=True)

            result = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-result))

            # Regularization Component
            reg_loss = self.l_w * tf.reduce_sum([tf.nn.l2_loss(gamma_u),
                                                 tf.nn.l2_loss(gamma_pos),
                                                 tf.nn.l2_loss(gamma_neg),
                                                 tf.nn.l2_loss(theta_u)]) \
                       + self.l_b * tf.nn.l2_loss(beta_pos) \
                       + self.l_b * tf.nn.l2_loss(beta_neg) / 10 \
                       + self.l_e * tf.reduce_sum([tf.nn.l2_loss(self.E), tf.nn.l2_loss(self.Bp)])

            # Loss to be optimized
            loss += reg_loss

        grads = t.gradient(loss, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp])
        self.optimizer.apply_gradients(zip(grads, [self.Bi, self.Gu, self.Gi, self.Tu, self.E, self.Bp]))

        return loss

    # @tf.function
    def predict_batch(self, start, stop, gi, bi, feat):
        return bi + tf.reduce_sum(self.Gu[start:stop] * gi, axis=1) \
               + tf.reduce_sum(self.Tu[start:stop] * tf.matmul(feat, self.E), axis=1) \
               + tf.squeeze(tf.matmul(feat, self.Bp))

    # @tf.function
    def predict_item_batch(self, start, stop, start_item, stop_item, feat):
        return self.Bi[start_item:(stop_item + 1)] + tf.matmul(self.Gu[start:stop], self.Gi[start_item:(stop_item + 1)],
                                                               transpose_b=True) \
               + tf.matmul(self.Tu[start:stop], tf.matmul(feat, self.E), transpose_b=True) \
               + tf.squeeze(tf.matmul(feat, self.Bp))

    def get_config(self):
        raise NotImplementedError

    # @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def _add_fake_user_by_item(self, item, image):

        ti = tf.matmul(self.feature_extractor(image, training=False), self.E)
        self.Tu = tf.concat([self.Tu, ti], axis=0)
        gi = self.Gi[item]
        self.Gu = tf.concat([self.Gu, tf.expand_dims(gi, axis=0)], axis=0)

        self._num_users += 1
        return self._num_users - 1

    def _remove_fake_user_by_item(self, size_random_items):

        self.Tu = self.Tu[:(self.Tu.shape[0] - size_random_items)]
        self.Gu = self.Gu[:(self.Tu.shape[0] - size_random_items)]

        self._num_users -= size_random_items

    def _score_users(self, next_eval_batch, us):

        scores = np.array([[0] * self._num_items] * len(us), dtype=np.float)
        for u_id, u in enumerate(us):
            for batch in next_eval_batch:
                item, feat = batch
                p = self.predict_item_batch(us[0], us[0] + 1, item[0], item[-1], tf.Variable(feat))
                scores[u_id][item] = p
        return scores

    def _score_users_item(self, image, item, us):

        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        feature_i = self.feature_extractor(image, training=False)

        scores = np.array([0] * len(us), dtype=np.float)
        for i, u in enumerate(us):
            scores[i] = (beta_i + tf.reduce_sum((self.Gu[u] * gamma_i), axis=0) + \
                         tf.reduce_sum((self.Tu[u] * tf.matmul(feature_i, self.E)), axis=1) + \
                         tf.squeeze(tf.matmul(feature_i, self.Bp)))[0]
        return scores

    def _score_user_item(self, image, item, u):

        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        feature_i = self.feature_extractor(image, training=False)

        score = (beta_i + tf.reduce_sum((self.Gu[u] * gamma_i), axis=0) + \
                 tf.reduce_sum((self.Tu[u] * tf.matmul(feature_i, self.E)), axis=1) + \
                 tf.squeeze(tf.matmul(feature_i, self.Bp)))[0]
        return score

    def _get_rank(self, u_scores, score):
        return int(np.searchsorted(-u_scores, -score)) + 1

    def _get_median_user(self, users, t, image, users_scores, percentile=0.25):
        ranks = []
        d = int(1 / percentile)
        for u in users:
            u_scores = users_scores[u]
            score = self._score_user_item(image, t, u)
            rank = self._get_rank(u_scores, score)
            ranks.append(rank)
        i = np.argsort(ranks)[len(ranks) // d]
        return users[i]

    def _solve(self, W, y):
        x = linear_model.LinearRegression().fit(W, y).coef_
        return x

    def bb_category_attack(self, original_image, eps=32, eps_step=12, steps=20, target_class=0):
        """
        Di Noia et al.
        Black Box attack on general Population
        """
        adv_x = projected_gradient_descent(self._ife, original_image, eps, eps_step, steps, np.inf, y=[target_class], targeted=True)
        adv_x = tf.clip_by_value(adv_x, [-103.939, -116.779, -123.68], [255 - 103.939, 255 - 116.779, 255 - 123.68])
        return adv_x

    def bb_rank_attack(self, next_eval_batch, attack_target_item, eps=32, steps=20):
        """
        cohen et al.
        Black Box attack on general Population (RANK)
        """
        random_users = random.sample(range(self._num_users), k=1000)  # k=1000

        _, item, image = attack_target_item
        steps = steps  # 20
        n_examples = 32  # 32
        gamma = 7
        zeros_initializer = tf.zeros_initializer()

        print(f'BB RANK on ITEM {item.numpy()[0]}')

        print('\tAdding fake_user_by_item')
        fake_users = []
        random_items = random.sample(range(self._num_users), k=200)  # k=200 in the paper
        for item in random_items:
            user = self._add_fake_user_by_item(item, image)
            fake_users.append(user)
        print('\tEND addition fake_user_by_item')

        print('\tEvaluating users_scores')
        users_scores = {}
        for us in [fake_users, random_users]:
            scores = self._score_users(next_eval_batch, us)
            item_scores = self._score_users_item(image, item, us)
            scores.sort(axis=-1)
            scores = np.flip(scores, axis=-1)
            ranks = []
            for u, u_scores, score in zip(us, scores, item_scores):
                users_scores[u] = u_scores
                rank = self._get_rank(u_scores, score)
                ranks.append(rank)
            ranks = np.array(ranks)
            print((ranks <= 20).mean())
        print('\tEND Evaluation users_scores')

        self.perturbation = tf.Variable(
            initial_value=zeros_initializer(shape=image.numpy().shape, dtype=tf.float32), trainable=True)
        perturbed_image = image + self.perturbation

        for step in range(steps):
            print(f'\t Step {step + 1}/{steps}')
            W, y = [], []
            feat_t = self.feature_extractor(perturbed_image)
            u = self._get_median_user(fake_users, item, perturbed_image, users_scores)
            u_scores = users_scores[u]
            y_t = self._score_user_item(perturbed_image, item, u).numpy()

            init_rank = self._get_rank(u_scores, y_t)

            for n_example in range(n_examples):
                print(f'\t n_examples {n_example + 1}/{n_examples}')
                d = np.random.choice(range(-gamma, gamma + 1), size=perturbed_image.shape)
                img_i = perturbed_image + d

                feat_i = self.feature_extractor(img_i, training=False)

                y_i = self._score_user_item(img_i, item, u)

                diff = feat_i - feat_t

                W.append(diff.numpy())
                y.append(y_t - y_i)

            x = self._solve(np.vstack(W), y)

            with tf.GradientTape() as t:
                loss = -x * self.feature_extractor(image + self.perturbation, training=False)

            print(f'LOSS: {np.sum(loss)}')
            grads = t.gradient(loss, [self.perturbation])
            # Here the SIGN Changing
            optimal_perturbation = tf.sign(grads[0])
            optimal_perturbation = tf.stop_gradient(optimal_perturbation)
            scaled_perturbation = tf.multiply(eps, optimal_perturbation)

            # Scale perturbation to be the solution for the norm=eps rather than norm=1 problem
            adv_x = tf.clip_by_value(self.original_target_image - tf.clip_by_value(perturbed_image + scaled_perturbation - image, -eps, eps), [-103.939, -116.779, -123.68], [255-103.939, 255-116.779, 255-123.68])
            self.perturbation.assign(self.original_target_image - adv_x)

            perturbed_image = image + self.perturbation
            score = self._score_user_item(perturbed_image, item, u)
            rank = self._get_rank(u_scores, score)

            print(f"Step: {step + 1}, median user={u}, {init_rank}->{rank}")

        # REMOVE FAKE USERS FROM THE MODEL
        # self._remove_fake_user_by_item(len(random_items)) # BUG HERE

        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        feature_i = self.feature_extractor(image + self.perturbation, training=False)
        attacked_scores = beta_i + tf.reduce_sum((self.Gu[:(self.Tu.shape[0] - len(random_items))] * gamma_i), axis=1) + \
                          tf.reduce_sum(
                              (self.Tu[:(self.Tu.shape[0] - len(random_items))] * tf.matmul(feature_i, self.E)),
                              axis=1) + \
                          tf.squeeze(tf.matmul(feature_i, self.Bp))

        return perturbed_image, attacked_scores

    def wb_sign_attack(self, original_image, eps_step=8, eps=32, steps=5):
        """
            Cohen et al.
            White Box Attack
        """
        zeros_initializer = tf.zeros_initializer()

        self.original_target_image = tf.Variable(initial_value=original_image, trainable=True)
        perturbation = tf.Variable(initial_value=zeros_initializer(shape=original_image.numpy().shape, dtype=tf.float32), trainable=False)
        # print(f'\t\t\tImg {item}')
        valid_loss = -np.inf
        best_perturbations = perturbation

        for _ in range(steps):

            with tf.GradientTape() as t:
                val = tf.reduce_sum(tf.reduce_sum((self.Tu * tf.matmul(self.feature_extractor(self.original_target_image - perturbation, training=False), self.E)), axis=1) + \
                          tf.squeeze(tf.matmul(self.feature_extractor(self.original_target_image - perturbation, training=False), self.Bp)))

                loss = - val

            grads = t.gradient(loss, [self.original_target_image])
            grads = tf.stop_gradient(grads)
            # Here the SIGN Changing
            optimal_perturbation = tf.sign(grads[0])
            scaled_perturbation = tf.multiply(eps_step, optimal_perturbation)

            # Scale perturbation to be the solution for the norm=eps rather than norm=1 problem
            adv_x = tf.clip_by_value(self.original_target_image - tf.clip_by_value(perturbation + scaled_perturbation, -eps, eps), [-103.939, -116.779, -123.68], [255-103.939, 255-116.779, 255-123.68])
            perturbation.assign(self.original_target_image - adv_x)

            if valid_loss <= val:
                valid_loss = val
                best_perturbations = perturbation
                # print('BEST')

        return self.original_target_image - best_perturbations

    def insa_attack(self, original_image, eps=32, steps=5):

        zeros_initializer = tf.zeros_initializer()

        self.original_target_image = tf.Variable(initial_value=original_image, trainable=True)
        perturbation = tf.Variable(initial_value=zeros_initializer(shape=original_image.numpy().shape, dtype=tf.float32), trainable=False)

        valid_loss = -np.inf
        best_perturbations = perturbation
        # print(f'\t\t\tImg {item}')
        for _ in range(steps):
            with tf.GradientTape() as t:
                val = tf.reduce_sum(tf.reduce_sum((self.Tu * tf.matmul(self.feature_extractor(self.original_target_image - perturbation, training=False), self.E)), axis=1) + \
                          tf.squeeze(tf.matmul(self.feature_extractor(self.original_target_image - perturbation, training=False), self.Bp)))

                loss = - val

            grads = t.gradient(loss, [self.original_target_image])
            grads = tf.stop_gradient(grads)[0]

            adv_x = tf.clip_by_value(self.original_target_image - tf.clip_by_value(perturbation + grads, -eps, eps), [-103.939, -116.779, -123.68], [255-103.939, 255-116.779, 255-123.68])
            perturbation.assign(self.original_target_image - adv_x)

            if valid_loss <= val:
                valid_loss = val
                best_perturbations = perturbation
                # print('BEST')

        return self.original_target_image - best_perturbations

    def score_items(self, batch, defense='nodefense'):
        item, original_images, attacked_images = batch

        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, item))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, item))
        if defense == 'denoiser':
            # Denoising The Image
            attacked_images = self._unet(attacked_images, training=True)
            attacked_feature_i = self.feature_extractor(attacked_images, training=False)

            original_images = self._unet(original_images, training=True)
            original_feature_i = self.feature_extractor(original_images, training=False)
        if defense == 'madry':
            # Pass through defended feature extractor
            attacked_feature_i = tf.expand_dims(np.squeeze(self._madry_feature_extractor.feature_model(
                torch.from_numpy(tf.reshape(attacked_images, (1, 3, 224, 224)).numpy())).data.cpu().numpy()), axis=0)
            original_feature_i = tf.expand_dims(np.squeeze(self._madry_feature_extractor.feature_model(
                torch.from_numpy(tf.reshape(original_images, (1, 3, 224, 224)).numpy())).data.cpu().numpy()), axis=0)
        else:
            attacked_feature_i = self.feature_extractor(attacked_images, training=False)
            original_feature_i = self.feature_extractor(original_images, training=False)

        attacked_scores = beta_i + tf.reduce_sum((self.Gu * gamma_i), axis=1) + \
                          tf.reduce_sum((self.Tu * tf.matmul(attacked_feature_i, self.E)), axis=1) + \
                          tf.squeeze(tf.matmul(attacked_feature_i, self.Bp))

        not_attacked_scores = beta_i + tf.reduce_sum((self.Gu * gamma_i), axis=1) + \
                              tf.reduce_sum((self.Tu * tf.matmul(
                                  original_feature_i, self.E)), axis=1) + \
                              tf.squeeze(tf.matmul(original_feature_i, self.Bp))

        return attacked_scores, not_attacked_scores

    def train_step_defense(self, batch, activate_rec_loss=False):
        items, original_images, attacked_images = batch
        with tf.GradientTape() as t:
            denoised_images = self._unet(attacked_images, training=True)

            beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, items))
            gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, items))

            # denoised_scores = beta_i + tf.reduce_sum((self.Gu * gamma_i), axis=1) + \
            #                   tf.reduce_sum((self.Tu * tf.matmul(self.feature_extractor(denoised_images, training=False), self.E)), axis=1) + \
            #                   tf.squeeze(tf.matmul(self.feature_extractor(denoised_images, training=False), self.Bp))

            denoised_scores = beta_i + tf.matmul(self.Gu, gamma_i, transpose_b=True) + \
                              tf.matmul(self.Tu, tf.matmul(
                                  self.feature_extractor(denoised_images, training=False), self.E), transpose_b=True) + \
                              tf.squeeze(tf.matmul(self.feature_extractor(denoised_images, training=False), self.Bp))

            # original_scores = beta_i + tf.reduce_sum((self.Gu * gamma_i), axis=1) + \
            #                   tf.reduce_sum((self.Tu * tf.matmul(self.feature_extractor(original_images, training=False), self.E)), axis=1) + \
            #                   tf.squeeze(tf.matmul(self.feature_extractor(original_images, training=False), self.Bp))

            original_scores = beta_i + tf.matmul(self.Gu, gamma_i, transpose_b=True) + \
                              tf.matmul(self.Tu, tf.matmul(
                                  self.feature_extractor(original_images, training=False), self.E), transpose_b=True) + \
                              tf.squeeze(tf.matmul(self.feature_extractor(original_images, training=False), self.Bp))

            # recommender_loss = tf.reduce_mean(tf.math.pow((original_scores - denoised_scores), 2))
            recommendation_loss = tf.reduce_mean(tf.math.pow((original_scores - denoised_scores), 2))
            feature_loss = tf.reduce_mean(tf.reduce_sum(
                tf.abs(self.feature_extractor(denoised_images, training=False) - self.feature_extractor(original_images,
                                                                                                        training=False)),
                axis=1))

            loss = feature_loss
            if activate_rec_loss:
                loss += recommendation_loss

        grads = t.gradient(loss, [self._unet.trainable_variables])
        self.optimizer.apply_gradients(zip(grads[0], self._unet.trainable_variables))

        return loss

    def call_defense(self, batch):
        items, original_images, attacked_images = batch
        denoised_images = self._unet(attacked_images, training=True)

        beta_i = tf.squeeze(tf.nn.embedding_lookup(self.Bi, items))
        gamma_i = tf.squeeze(tf.nn.embedding_lookup(self.Gi, items))

        denoised_scores = beta_i + tf.matmul(self.Gu, gamma_i, transpose_b=True) + \
                          tf.matmul(self.Tu, tf.matmul(
                              self.feature_extractor(denoised_images, training=False), self.E), transpose_b=True) + \
                          tf.squeeze(tf.matmul(self.feature_extractor(denoised_images, training=False), self.Bp))

        original_scores = beta_i + tf.matmul(self.Gu, gamma_i, transpose_b=True) + \
                          tf.matmul(self.Tu, tf.matmul(
                              self.feature_extractor(original_images, training=False), self.E), transpose_b=True) + \
                          tf.squeeze(tf.matmul(self.feature_extractor(original_images, training=False), self.Bp))

        loss = tf.reduce_mean(tf.math.pow((original_scores - denoised_scores), 2)) + tf.reduce_mean(tf.reduce_sum(
            tf.abs(self.feature_extractor(denoised_images, training=False) - self.feature_extractor(original_images,
                                                                                                    training=False)),
            axis=1))
        return loss
