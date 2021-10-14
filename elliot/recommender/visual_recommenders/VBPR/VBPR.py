"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import shutil
import pickle
import torch
from PIL import Image
import multiprocessing as mp

from elliot.dataset.samplers import pairwise_pipeline_sampler_vbpr as ppsv
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.visual_recommenders.VBPR.VBPR_model import VBPR_model
from elliot.utils.write import store_recommendation
import ast
import pandas as pd
import gc

np.random.seed(0)
tf.random.set_seed(0)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

target_hr = []
original_hr = []


class VBPR(RecMixin, BaseRecommenderModel):
    r"""
    VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback

    For further details, please refer to the `paper <http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11914>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        factors_d: Dimension of visual factors
        batch_size: Batch size
        l_w: Regularization coefficient
        l_b: Regularization coefficient of bias
        l_e: Regularization coefficient of projection matrix

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        VBPR:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          factors: 100
          factors_d: 20
          batch_size: 128
          l_w: 0.000025
          l_b: 0
          l_e: 0.002
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random

        self._params_list = [
            ("_batch_eval", "batch_eval", "be", 512, int, None),
            ("_factors", "factors", "factors", 100, None, None),
            ("_factors_d", "factors_d", "factors_d", 20, None, None),
            ("_learning_rate", "lr", "lr", 0.0005, None, None),
            ("_l_w", "l_w", "l_w", 0.000025, None, None),
            ("_l_b", "l_b", "l_b", 0, None, None),
            ("_l_e", "l_e", "l_e", params.l_w, None, None)
        ]
        self.autoset_params()

        # TODO
        # Crate Attack Dataset
        self._create_attack_dataset = list(ast.literal_eval(self._params.meta.create_attack_dataset))
        # Train Defendant
        self._train_defendant = self._params.meta.train_defendant
        self._epoch_train_defendant = self._params.epoch_train_defendant
        self._lr_train_defendant = self._params.lr_train_defendant
        self._batch_train_defendant = self._params.batch_train_defendant
        # Attack Strategy
        self._attack = self._params.meta.attack
        self._test_attack_strategies = list(ast.literal_eval(self._params.meta.test_attack_strategies))
        self._bb_category_class = self._params.meta.bb_category_class
        # List of Target Items
        self._num_attacked_items = self._params.meta.num_attacked_items
        self._adversarial_top_k = self._params.meta.adversarial_top_k
        # Defense Strategy
        self._defenses = list(self._params.meta.defense.replace('(', '').replace(')', '').split(','))
        # self._evaluate_with_denoiser = self._params.meta.evaluate_with_denoiser

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        item_indices = [self._data.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._sampler = ppsv.Sampler(self._data.i_train_dict,
                                     item_indices,
                                     self._data.side_information_data.visual_feature_path,
                                     self._epochs)

        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        self._model = VBPR_model(self._factors,
                                 self._factors_d,
                                 self._learning_rate,
                                 self._l_w,
                                 self._l_b,
                                 self._l_e,
                                 self._data.visual_features_shape,
                                 self._num_users,
                                 self._num_items,
                                 self._lr_train_defendant,
                                 self._defenses)

        # only for evaluation purposes
        self._next_eval_batch = self._sampler.pipeline_eval(self._batch_eval)

    @property
    def name(self):
        return "VBPR" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def start_train(self):
        best_metric_value = 0
        loss = 0
        steps = 0
        it = 0
        early_stopping = 5

        with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
            for batch in self._next_batch:
                steps += 1
                loss += self._model.train_step(batch)
                t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                t.update()

                # epoch is over
                if steps == self._data.transactions // self._batch_size:
                    t.reset()
                    if not (it + 1) % self._validation_rate:
                        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                        result_dict = self.evaluator.eval(recs)
                        self._results.append(result_dict)

                        self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / steps:.3f}')

                        if self._results[-1][self._validation_k]["val_results"][
                            self._validation_metric] > best_metric_value:
                            best_recs = recs
                            best_metric_value = self._results[-1][self._validation_k]["val_results"][
                                self._validation_metric]
                            early_stopping = 5
                            if self._save_weights:
                                self._model.save_weights(self._saving_filepath)
                            if self._save_recs:
                                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")
                        else:
                            early_stopping -= 1
                            if early_stopping == 0:
                                print('Reached Early Stopping Condition at Epoch {0}\n\tEXIT'.format(it + 1))
                                break

                    it += 1
                    steps = 0
                    loss = 0
        return best_recs

    def train(self):
        is_restored = False

        if self._restore:
            is_restored, best_recs = self.restore_weights()

        if self._restore is False or is_restored is False:
            print('This Model will start the training!')
            best_recs = self.start_train()

        if self._attack:
            self.attack_pipeline(best_recs)

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            print('\tUser {0}\{1}'.format(offset_stop, self._num_users))
            predictions = np.empty((offset_stop - offset, self._num_items))
            for batch in self._next_eval_batch:
                item, feat = batch
                # print('\t\tItem {0}\{1}'.format(item[-1], self._num_items))
                p = self._model.predict_item_batch(offset, offset_stop,
                                                   item[0], item[-1],
                                                   tf.Variable(feat))
                predictions[:(offset_stop - offset), item] = p
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k

    def optimized_get_defense_recommendations(self, k: int = 100, defense: str = 'nodefense'):
        print(f'Get defended recommendation on {defense}')
        batch_eval = 512

        self._next_defense_eval_batch = self._sampler.pipeline_defense_eval(
            self._data.side_information_data.images_src_folder,
            self._data.output_image_size, 1)

        # Save the denoised features into a directory

        print('*** Start the Attack/Defense Experiments ***')
        image_folder = self._data.side_information_data.images_src_folder.replace('images/', '')
        feat_test_attacked_images_src_folder = os.path.join(image_folder, 'DenoisedFeature_Test_' + self.name)

        if os.path.exists(feat_test_attacked_images_src_folder):
            shutil.rmtree(feat_test_attacked_images_src_folder)
        os.makedirs(feat_test_attacked_images_src_folder)

        start_time = time.time()
        for batch in self._next_defense_eval_batch:
            item, image = batch
            if defense == 'denoiser':
                denoised = self._model._unet(image, training=False)
                feat = self._model.feature_extractor(denoised)
                save_np(feat[0], f'{feat_test_attacked_images_src_folder}/{item[0]}')

            if (item[0]+1) % 100 == 0:
                print(f'{int(item[0])+1}/{self._num_items} in {round(time.time() - start_time, 2)}')
                start_time = time.time()

        self._next_defense_eval_batch_by_feat = self._sampler.pipeline_defense_eval_by_feat(
            feat_test_attacked_images_src_folder + '/',
            self._data.output_image_size, batch_eval)

        predictions_top_k = {}
        start_time = time.time()
        for index, offset in enumerate(range(0, self._num_users, batch_eval)):
            offset_stop = min(offset + batch_eval, self._num_users)
            predictions = np.empty((offset_stop - offset, self._num_items))
            for batch in self._next_defense_eval_batch_by_feat:
                item, feat = batch
                # print('\t\tItem {0}\{1}'.format(item[-1], self._num_items))
                p = self._model.predict_item_batch(offset, offset_stop,
                                                   item[0], item[-1],
                                                   tf.Variable(feat))
                predictions[:(offset_stop - offset), item] = p
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
            print(f'\tUser {offset_stop} of {self._num_users} in {round(time.time() - start_time, 2)}')
            start_time = time.time()

        return predictions_top_k

    def get_defense_recommendations(self, k: int = 100, defense: str = 'nodefense'):
        print(f'Get defended recommendation on {defense}')
        batch_eval = 600

        self._next_defense_eval_batch = self._sampler.pipeline_defense_eval(
            self._data.side_information_data.images_src_folder,
            self._data.output_image_size, batch_eval)

        predictions_top_k = {}
        start_time = time.time()
        for index, offset in enumerate(range(0, self._num_users, batch_eval)):
            offset_stop = min(offset + batch_eval, self._num_users)
            predictions = np.empty((offset_stop - offset, self._num_items))
            for batch in self._next_defense_eval_batch:
                item, image = batch
                if defense == 'madry':
                    feat = np.squeeze(self._model._madry_feature_extractor.feature_model(torch.from_numpy(
                        (tf.reshape(image, (
                            image.shape[0], image.shape[3], image.shape[1],
                            image.shape[2])).numpy()))).data.cpu().numpy())
                elif defense == 'denoiser':
                    denoised = self._model._unet(image, training=False)
                    feat = self._model.feature_extractor(denoised)
                # print('\t\tItem {0}\{1}'.format(item[-1], self._num_items))
                p = self._model.predict_item_batch(offset, offset_stop,
                                                   item[0], item[-1],
                                                   tf.Variable(feat))
                predictions[:(offset_stop - offset), item] = p
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
            print(f'\tUser {offset_stop} of {self._num_users} in {round(time.time() - start_time, 2)}')
            start_time = time.time()

        return predictions_top_k

    def create_train(self, train_attacked_images_src_folder, training_images):
        print('Creating Training Datasets of Attacked Images...')
        # Create Dataset of Attacked Images
        if os.path.exists(train_attacked_images_src_folder):
            shutil.rmtree(train_attacked_images_src_folder)
        os.makedirs(train_attacked_images_src_folder)

        for image_batch in self._sampler.pipeline_attack_no_tf(training_images,
                                                               self._data.side_information_data.images_src_folder,
                                                               self._data.output_image_size):
            num_image_batch, train_image, original_image = image_batch
            print(f'\tTrain Image {num_image_batch + 1}/{len(training_images)}')
            item_id = str(train_image)
            if os.path.exists(os.path.join(train_attacked_images_src_folder, item_id)):
                shutil.rmtree(os.path.join(train_attacked_images_src_folder, item_id))
            os.makedirs(os.path.join(train_attacked_images_src_folder, item_id))

            ## AIP (INSA)
            if self._test_attack_strategies[2]:
                for step in [1, 2, 8]:
                    for eps in np.random.randint(1, high=16, size=1):
                        start = time.time()
                        with open(
                                os.path.join(train_attacked_images_src_folder, item_id,
                                             f'insa-eps{eps}-step{step}.npy'),
                                'wb') as f:
                            attacked_images = self._model.insa_attack(original_image, eps, step)
                            np.save(f, np.array(attacked_images[0]))
                            del attacked_images
                            gc.collect()
                        print(f'\t\tINSA (Liu et al.) Eps: {eps} Steps: {step} in {round(time.time() - start, 2)}')

            ## Cohen et al. (WB-SIGN)
            if self._test_attack_strategies[3]:
                for step in [1, 2, 8]:
                    for eps in np.random.randint(1, high=16, size=1):
                        start = time.time()
                        eps_step = 2.5 * eps / step if 2.5 * eps / step < eps else eps
                        with open(os.path.join(train_attacked_images_src_folder, item_id,
                                               f'wbsign-eps{eps}-step{step}.npy'), 'wb') as f:
                            attacked_images = self._model.wb_sign_attack(original_image, eps_step, eps, step)
                            np.save(f, np.array(attacked_images[0]))
                            del attacked_images
                            gc.collect()
                        print(
                            f'\t\tSIGN (Cohen et al.) Eps-Step: {eps_step}/{eps} Steps: {step} in {round(time.time() - start, 2)}')

    def create_valid(self, valid_attacked_images_src_folder, valid_images):
        # We can add also other attacks below
        # Create the validation dataset
        print('Creating Validation Datasets of Attacked Images...')
        if os.path.exists(valid_attacked_images_src_folder):
            shutil.rmtree(valid_attacked_images_src_folder)
        os.makedirs(valid_attacked_images_src_folder)

        for image_batch in self._sampler.pipeline_attack_no_tf(valid_images,
                                                               self._data.side_information_data.images_src_folder,
                                                               self._data.output_image_size):
            num_image_batch, valid_image, original_image = image_batch
            print(f'\tValid Image {num_image_batch + 1}/{len(valid_images)}')
            item_id = str(valid_image)
            if os.path.exists(os.path.join(valid_attacked_images_src_folder, item_id)):
                shutil.rmtree(os.path.join(valid_attacked_images_src_folder, item_id))
            os.makedirs(os.path.join(valid_attacked_images_src_folder, item_id))

            # White Box Attack
            ## AIP (INSA)
            if self._test_attack_strategies[2]:
                for step in [1, 2, 4]:
                    for eps in np.random.randint(1, high=16, size=1):
                        start = time.time()
                        with open(
                                os.path.join(valid_attacked_images_src_folder, item_id,
                                             f'insa-eps{eps}-step{step}.npy'),
                                'wb') as f:
                            attacked_images = self._model.insa_attack(original_image, eps, step)
                            np.save(f, np.array(attacked_images[0]))
                            del attacked_images
                            gc.collect()
                        print(f'\t\tINSA (Liu et al.) Eps: {eps} Steps: {step} in {round(time.time() - start, 2)}')

            ## Cohen et al. (WB-SIGN)
            if self._test_attack_strategies[3]:
                for step in [1, 2, 4]:
                    for eps in np.random.randint(1, high=16, size=1):
                        start = time.time()
                        eps_step = 2.5 * eps / step if 2.5 * eps / step < eps else eps
                        with open(os.path.join(valid_attacked_images_src_folder, item_id,
                                               f'wbsign-eps{eps}-step{step}.npy'), 'wb') as f:
                            attacked_images = self._model.wb_sign_attack(original_image, eps_step, eps, step)
                            np.save(f, np.array(attacked_images[0]))
                            del attacked_images
                            gc.collect()
                        print(
                            f'\t\tSIGN (Cohen et al.) Eps-Step: {eps_step}/{eps} Steps: {step} in {round(time.time() - start, 2)}')

    def create_test(self, test_attacked_images_src_folder, test_images):
        # Create the test dataset with other attacks (Black Box)
        print('Creating Test Datasets of Attacked Images...')
        if os.path.exists(test_attacked_images_src_folder):
            shutil.rmtree(test_attacked_images_src_folder)
        os.makedirs(test_attacked_images_src_folder)

        for image_batch in self._sampler.pipeline_attack_no_tf(test_images,
                                                               self._data.side_information_data.images_src_folder,
                                                               self._data.output_image_size):
            num_image_batch, test_image, original_image = image_batch

            print(f'\tTest Image {num_image_batch + 1}/{len(test_images)}')
            item_id = str(test_image)
            # _, x = image_batch
            # tf.keras.applications.resnet.preprocess_input(x)
            if os.path.exists(os.path.join(test_attacked_images_src_folder, item_id)):
                shutil.rmtree(os.path.join(test_attacked_images_src_folder, item_id))
            os.makedirs(os.path.join(test_attacked_images_src_folder, item_id))

            # Black Box Attacks
            ## DI NOIA et al.
            if self._test_attack_strategies[0]:
                for eps in [4, 8, 16]:
                    for step in [1, 4, 8]:
                        start = time.time()
                        eps_step = 2.5 * eps / step if 2.5 * eps / step < eps else eps
                        with open(os.path.join(test_attacked_images_src_folder, item_id,
                                               f'bbcat{self._bb_category_class}-eps{eps}-step{step}.npy'), 'wb') as f:
                            attacked_images = self._model.bb_category_attack(original_image, eps, eps_step, step,
                                                                             self._bb_category_class)
                            np.save(f, np.array(attacked_images[0]))
                            del attacked_images
                            gc.collect()
                        print(f'\t\tTAaMR (Di Noia et al.) Eps: {eps} Steps: {step} in {round(time.time() - start, 2)}')

            ## COHEN et al.
            if self._test_attack_strategies[1]:
                attacked_images, _ = self._model.bb_rank_attack(self._next_eval_batch, image_batch, eps, step)
                np.save(os.path.join(test_attacked_images_src_folder, item_id, f'bbrank-eps{eps}-step{step}.jpg'),
                        np.array(attacked_images[0]))
                self._model.load_weights(self._saving_filepath)
                print('\t\tBB-Cohen Completed')

            # White Box Attacks
            ## AIP (INSA)
            if self._test_attack_strategies[2]:
                for eps in [4, 8, 16]:
                    for step in [1, 4, 8]:
                        start = time.time()
                        with open(
                                os.path.join(test_attacked_images_src_folder, item_id, f'insa-eps{eps}-step{step}.npy'),
                                'wb') as f:
                            attacked_images = self._model.insa_attack(original_image, eps, step)
                            np.save(f, np.array(attacked_images[0]))
                            del attacked_images
                            gc.collect()
                        print(f'\t\tINSA (Liu et al.) Eps: {eps} Steps: {step} in {round(time.time() - start, 2)}')

            ## Cohen et al. (WB-SIGN)
            if self._test_attack_strategies[3]:
                for eps in [4, 8, 16]:
                    for step in [1, 4, 8]:
                        start = time.time()
                        eps_step = 2.5 * eps / step if 2.5 * eps / step < eps else eps
                        with open(os.path.join(test_attacked_images_src_folder, item_id,
                                               f'wbsign-eps{eps}-step{step}.npy'), 'wb') as f:
                            attacked_images = self._model.wb_sign_attack(original_image, eps_step, eps, step)
                            np.save(f, np.array(attacked_images[0]))
                            del attacked_images
                            gc.collect()
                        print(
                            f'\t\tSIGN (Cohen et al.) Eps-Step: {eps_step}/{eps} Steps: {step} in {round(time.time() - start, 2)}')

            del image_batch
            gc.collect()

    def format_attack_results(self, dict_glb_original_hr, dict_glb_target_hr, def_name):
        if def_name == 'denoiser':
            name = self._config.path_output_rec_result + f"attack-{def_name}-lr{self._lr_train_defendant}-bs{self._batch_train_defendant}.tsv"
        elif def_name == 'madry':
            name = self._config.path_output_rec_result + f"attack-{def_name}.tsv"  # For now equal to no-defense but we can add parameters to MADRY
        else:
            name = self._config.path_output_rec_result + f"attack-{def_name}.tsv"

        with open(name, "w") as f:
            f.write(
                f'Defense\tItem\tAttack\tEpsilon\tSteps\tHRBefAtt@{str(self._adversarial_top_k)}\tHRAftAtt@{str(self._adversarial_top_k)}\n')
            # dict_glb_target_hr[def_name][attack_name][eps_name][step_name]
            for defense_name in dict_glb_target_hr.keys():
                for item_name in dict_glb_target_hr[defense_name].keys():
                    for attack_name in dict_glb_target_hr[defense_name][item_name].keys():
                        for eps_name in dict_glb_target_hr[defense_name][item_name][attack_name].keys():
                            for step_name in dict_glb_target_hr[defense_name][item_name][attack_name][eps_name].keys():
                                result_after_attack = \
                                    dict_glb_target_hr[defense_name][item_name][attack_name][eps_name][step_name]
                                result_before_attack = \
                                    dict_glb_original_hr[defense_name][item_name][attack_name][eps_name][step_name]
                                f.write(
                                    f'{defense_name}\t{item_name}\t{attack_name}\t{eps_name}\t{step_name}\t{str(np.mean(result_before_attack))}\t{str(np.mean(result_after_attack))}\n')

    def train_defense(self, train_attacked_images_src_folder, valid_attacked_images_src_folder):
        start = time.time()
        validation_loss = np.inf
        training_loss = np.inf
        print(f'Training Defense for {self._epoch_train_defendant} epochs...')
        # Training Defense (The pipeline has to read the adversarial perturbed images)
        for epoch in range(self._epoch_train_defendant):
            start_epoch = time.time()
            epoch_loss = []
            self._next_train_defense_batch = self._sampler.pipeline_defense(
                train_attacked_images_src_folder, self._data.side_information_data.images_src_folder,
                self._data.output_image_size, self._batch_train_defendant)
            for train_defense_batch in self._next_train_defense_batch:
                item, original_images, attacked_images = train_defense_batch
                batch_loss = self._model.train_step_defense((item, original_images, attacked_images),
                                                            activate_rec_loss=epoch >= self._epoch_train_defendant / 2)
                epoch_loss.append(batch_loss.numpy())

            # Save the weights on the validation dataset
            new_validation_loss = self.validate_defense(valid_attacked_images_src_folder)
            # new_validation_loss = self.validate_defense(train_attacked_images_src_folder)

            new_training_loss = np.mean(epoch_loss)
            if new_training_loss <= training_loss:
                training_loss = new_training_loss

            if (new_validation_loss < validation_loss) or ((new_validation_loss == validation_loss) and (new_training_loss <= training_loss)):
                validation_loss = new_validation_loss
                self._model._unet.save_weights(
                    self._saving_filepath + f'_defense-lr{self._lr_train_defendant}-bs{self._batch_train_defendant}')

            print(
                f'\tTrain Defendant Epoch {epoch + 1}-{self._epoch_train_defendant} - Defense Loss {np.mean(epoch_loss)} - Validation loss {validation_loss} - Rec. Component {epoch >= self._epoch_train_defendant / 2} - Time {(time.time() - start_epoch) / 60} minutes.')
        print(f'Trained in {(time.time() - start) / 60} minutes. ')

    def test_attack(self, original_recs, test_attacked_images_src_folder, items_id_by_popularity):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        name_file = self._config.path_output_rec_performance + f"attack-vbpr-lr{self._lr_train_defendant}-bs{self._batch_train_defendant}-{dt_string}.tsv"

        with open(name_file, 'w') as f:
            f.write(
                f'Defense\tItem\tPopularity\tAttack\tEps\tSteps\tHRBefAtt@{self._adversarial_top_k}\tHRAftAtt@{self._adversarial_top_k}\tNumImprov\tPredShift\n')

            for def_name in self._defenses:  # [0,1]
                recs = original_recs if def_name == 'nodefense' else self.optimized_get_defense_recommendations(
                    self.evaluator.get_needed_recommendations(), def_name)

                start = time.time()
                for test_defense_batch in self._sampler.pipeline_test_defense_no_tf(test_attacked_images_src_folder,
                                                                                    self._data.side_information_data.images_src_folder,
                                                                                    self._data.output_image_size):
                    name, target_item_id, original_images, attacked_images, im_num, tot_im_num = test_defense_batch
                    attacked_scores, original_score = self._model.score_items(
                        (target_item_id, original_images, attacked_images), defense=def_name)
                    # Evaluate the HR before and after the attack for the selected target item
                    global target_hr
                    global original_hr

                    target_hr = [0] * len(recs.keys())
                    original_hr = [0] * len(target_hr)
                    pool = mp.Pool(processes=12)

                    for user in recs.keys():
                        pool.apply_async(evaluate_prediction,
                                         args=(
                                         recs[user], user, self._adversarial_top_k, attacked_scores, target_item_id,),
                                         callback=store_predictions)

                    pool.close()
                    pool.join()

                    attack_name = name.split('-')[1]
                    eps_name = name.split('-')[2].replace('eps', '')
                    step_name = name.split('-')[3].replace('step', '')

                    if (im_num+1) % 10 == 0:
                        print(f'\t\t{im_num+1}/{tot_im_num} in {round(time.time() - start, 2)} secs.')
                        start = time.time()


                    f.write(
                        f'{def_name}\t{target_item_id}\t{items_id_by_popularity.tolist().index(target_item_id) + 1}\t{attack_name}\t{eps_name}\t{step_name}\t{np.mean(original_hr)}\t{np.mean(target_hr)}\t{np.mean(attacked_scores > original_score)}\t{np.mean(attacked_scores - original_score)}\n')

                if def_name == 'denoiser':
                    print('***** STORING FULL RESULTS ******')
                    # Store Defended Performance
                    denoised_results = self.evaluator.eval(recs)
                    name_file_recomm = self._config.path_output_rec_performance + f"denoised-vbpr-overall-performance-lr{self._lr_train_defendant}-bs{self._batch_train_defendant}-{dt_string}.tsv"
                    # print(denoised_results)
                    with open(name_file_recomm, 'w') as ff:
                        l = 'k\t'
                        for metric_name in denoised_results[list(denoised_results.keys())[0]]['test_results'].keys():
                            l += f'{metric_name}\t'
                        ff.write(l.strip()+'\n')
                        for k in denoised_results.keys():
                            line = f'{k}\t'
                            for metric in denoised_results[k]['test_results'].keys():
                                line += f'{denoised_results[k]["test_results"][metric]}\t'
                            line = line.strip()
                            ff.write(line + '\n')
                    print('***** END STORING FULL RESULTS ******')

    def attack_pipeline(self, recs):
        print('*** Start the Attack/Defense Experiments ***')
        attack_folder = self._data.side_information_data.images_src_folder.replace('images/', '') + 'attacks/'

        train_attacked_images_src_folder = os.path.join(attack_folder, 'Train_' + self.name)
        valid_attacked_images_src_folder = os.path.join(attack_folder, 'Valid_' + self.name)
        test_attacked_images_src_folder = os.path.join(attack_folder, 'Test_' + self.name)

        popularity = [0] * self._num_items
        for user in self._data.train_dict.keys():
            for item in self._data.train_dict[user].keys():
                popularity[item] += 1
        items_id_by_popularity = np.array(popularity).argsort()[::-1]
        images = np.random.choice(self._data.items, size=self._num_attacked_items,
                                  replace=False)  # Fixed with the Random Seed

        valid_threshold = 0
        # if self._create_attack_dataset == [1, 1, 1]:
        #     # CREATE ATTACK FOLDER
        #     print('Replacing the Attacks Folder With a New One')
        #     if os.path.exists(attack_folder):
        #         shutil.rmtree(attack_folder)
        #     os.makedirs(attack_folder)

        if self._create_attack_dataset[0]:
            train_threshold = int(self._num_attacked_items * 0.8)
            self.create_train(train_attacked_images_src_folder, images[:train_threshold])

        if self._create_attack_dataset[1]:
            valid_threshold = int(self._num_attacked_items * 0.9)
            self.create_valid(valid_attacked_images_src_folder,
                              images[train_threshold:valid_threshold])

        if self._create_attack_dataset[2]:
            self.create_test(test_attacked_images_src_folder, images[valid_threshold:])

        if 'denoiser' in list(self._defenses):
            if self._train_defendant:
                self.train_defense(train_attacked_images_src_folder, valid_attacked_images_src_folder)

            self._model._unet.load_weights(
                self._saving_filepath + f'_defense-lr{self._lr_train_defendant}-bs{self._batch_train_defendant}')
            print('Restored the UNet Denoiser with the best validation performance!')

        print('\t*** Evaluating the Attack/Defense Experiments ***')
        self.test_attack(recs, test_attacked_images_src_folder, items_id_by_popularity)
        print('\t*** End of Attack/Defense Experiments ***')

    def restore_weights(self):
        try:
            recs = None
            self._model.load_weights(self._saving_filepath)
            print(f"Model correctly Restored")

            try:
                print('Try to restore rec lists')
                recs = self.restore_recommendation(path=self._config.path_output_rec_result + f"{self.name}.tsv")
                print('Rec lists correctly Restored')
            except Exception as error:
                print(f'** Error in Try to restore rec lists\n\t{error}\n')
                print('Evaluate rec lists')
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())

            result_dict = self.evaluator.eval(recs)
            self._results.append(result_dict)

            print("******************************************")
            if self._save_recs:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")

            return True, recs

        except Exception as ex:
            print(f"Error in model restoring method! {ex}")
            return False, None  # Da cambiare in False

    def validate_defense(self, valid_attacked_images_src_folder):
        epoch_loss = []
        self._next_valid_defense_batch = self._sampler.pipeline_defense(valid_attacked_images_src_folder,
                                                                        self._data.side_information_data.images_src_folder,
                                                                        self._data.output_image_size, 2)
        for valid_defense_batch in self._next_valid_defense_batch:
            item, original_images, attacked_images = valid_defense_batch
            batch_loss = self._model.call_defense((item, original_images, attacked_images))
            epoch_loss.append(batch_loss.numpy())

        return np.mean(epoch_loss)

    def restore_recommendation(self, path=""):
        """
        Store recommendation list (top-k)
        :return:
        """
        recommendations = {}
        with open(path, 'r') as fin:
            while True:
                line = fin.readline().strip().split('\t')
                if line[0] == '':
                    break
                u = int(line[0])
                i = int(line[1])
                r = float(line[2])

                if u not in recommendations:
                    recommendations[u] = []
                recommendations[u].append((i, r))

        return recommendations


def evaluate_prediction(rec_list, user, adversarial_top_k, attacked_scores, target_item_id):
    t, o = 0, 0
    check_target = 0
    check_original = 0
    for k, (pred_item, pred_score) in enumerate(rec_list):
        if k < adversarial_top_k and not (check_target == 1 and check_original == 1):
            if attacked_scores[user].numpy() >= pred_score and check_target == 0:
                t = 1
                check_target = 1
            if target_item_id == pred_item and check_original == 0:
                o = 1
                check_original = 1
        else:
            break
    return t, o, user


def store_predictions(r):
    t, o, user = r
    global target_hr
    global original_hr

    target_hr[user] = t
    original_hr[user] = o

def save_np(npy, filename):
    """
    Store numpy to memory.
    Args:
        npy: numpy to save
        filename (str): filename
    """
    np.save(filename, npy)