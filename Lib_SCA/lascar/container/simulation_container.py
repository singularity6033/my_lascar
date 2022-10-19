# This file is part of lascar
#
# lascar is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#
# Copyright 2018 Manuel San Pedro, Victor Servant, Charles Guillemet, Ledger SAS - manuel.sanpedro@ledger.fr, victor.servant@ledger.fr, charles@ledger.fr

"""
simulation_container.py
"""
import random

import numpy as np
from tqdm import tqdm

from Lib_SCA.lascar.tools.aes import sbox, Aes
from Lib_SCA.lascar.tools.leakage_model import HammingPrecomputedModel
from Lib_SCA.config_extractor import TraceConfig
from .container import AbstractContainer, Trace

DEFAULT_KEY = [i for i in range(16)]


class BasicAesSimulationContainer(AbstractContainer):
    """
    BasicAesSimulationContainer is an AbstractContainer used to naively
    simulate traces during the first Subbyte of the first round of an AES.
    """

    def __init__(
            self,
            number_of_traces,
            noise=1,
            key=DEFAULT_KEY,
            seed=1337,
            leakage_model="default",
            additional_time_samples=10,
            **kwargs
    ):
        """
        Basic constructor:

        :param number_of_traces:
        :param noise: noise level of the simulated noise. (The noise follows a normal law with mean zero, and std=noiselevel)
        :param key: optional, the key used during AES first round (16 bytes)
        :param seed: optional, the seed used to generate values and noise (for reproductibility)
        :param leakage_model: optional, leakage model used (HammingWeight by default)
        :param additional_time_samples: optional: add dummy time samples
        """

        if leakage_model == "default":
            leakage_model = HammingPrecomputedModel()

        self.seed = seed
        self.noise = noise

        self.leakage_model = leakage_model
        self.additional_time_samples = additional_time_samples

        self.value_dtype = np.dtype(
            [("plaintext", np.uint8, (16,)), ("key", np.uint8, (16,)), ]
        )

        self.key = key

        AbstractContainer.__init__(self, number_of_traces, **kwargs)

        self.logger.debug("Creating BasicAesSimulationContainer.")
        self.logger.debug("Noise set to %f, seed set to %d.", self.noise, self.seed)
        self.logger.debug("Key set to %s", str(key))

    def generate_trace(self, idx):
        np.random.seed(seed=self.seed ^ idx)  # for reproducibility

        value = np.zeros((), dtype=self.value_dtype)
        value["plaintext"] = np.random.randint(0, 256, (16,), np.uint8)
        value["key"] = self.key

        leakage = np.random.normal(0, self.noise, (16 + self.additional_time_samples,))
        leakage[:16] += np.array(
            [
                self.leakage_model(sbox[value["plaintext"][i] ^ value["key"][i]])
                for i in range(16)
            ]
        )
        return Trace(leakage, value)


class AesSimulationContainer(AbstractContainer):
    """
    AesSimulationContainer is an AbstractContainer used to 
    simulate traces during all the round function of an AES.
    """

    def __init__(
            self,
            number_of_traces,
            noise=1,
            key=DEFAULT_KEY,
            seed=1337,
            leakage_model="default",
            additional_time_samples=10,
            **kwargs
    ):
        """
        Basic constructor:

        :param number_of_traces:
        :param noise: noise level of the simulated noise. (The noise follows a normal law with mean zero, and std=noiselevel)
        :param key: optional, the key used during AES first round (16 bytes)
        :param seed: optional, the seed used to generate values and noise (for reproductibility)
        :param leakage_model: optional, leakage model used (HammingWeight by default)
        :param additional_time_samples: optional: add dummy time samples
        """

        if leakage_model == "default":
            leakage_model = HammingPrecomputedModel()

        self.seed = seed
        self.noise = noise

        self.leakage_model = leakage_model
        self.additional_time_samples = additional_time_samples

        self.value_dtype = np.dtype(
            [
                ("plaintext", np.uint8, (16,)),
                ("key", np.uint8, (len(key),)),
                ("ciphertext", np.uint8, (16,)),
            ]
        )

        self.key = key

        AbstractContainer.__init__(self, number_of_traces, **kwargs)

        self.logger.debug("Creating AesSimulationContainer.")
        self.logger.debug("Noise set to %f, seed set to %d.", self.noise, self.seed)
        self.logger.debug("Key set to %s", str(key))

    def generate_trace(self, idx):
        np.random.seed(seed=self.seed ^ idx)  # for reproducibility

        value = np.zeros((), dtype=self.value_dtype)
        value["plaintext"] = np.random.randint(0, 256, (16,), np.uint8)
        value["key"] = self.key

        leakage = Aes.encrypt_keep_iv(value["plaintext"], Aes.key_schedule(value["key"]))
        value["ciphertext"] = leakage[-16:]

        leakage = np.array([self.leakage_model(i) for i in leakage])  # leakage model
        leakage = leakage + np.random.normal(0, self.noise, (len(leakage),))  # noise

        return Trace(leakage, value)


class SimulatedPowerTraceContainer(AbstractContainer):

    def __init__(self, number_of_traces=None, config_file_name='normal_simulated_traces.yaml', seed=1337, **kwargs):
        params = TraceConfig().get_config(config_file_name)
        if number_of_traces:
            self.number_of_trace = number_of_traces
        else:
            self.number_of_traces = params['number_of_traces']
        self.number_of_bytes = params['number_of_bytes']
        self.number_of_time_samples = params['number_of_time_samples']
        self.attack_sample_point = params['attack_sample_point']
        self.linear_coefficient_exp = float(params['linear_coefficient_exp'])
        self.linear_coefficient_switch = float(params['linear_coefficient_switch'])
        self.idx_exp = params['idx_exploitable_bytes']
        self.idx_switch = params['idx_switching_noise_bytes']
        if not len(self.idx_exp) + len(self.idx_switch) == self.number_of_bytes:
            print('[INFO] total number of exploitable signal bytes and switching noise bytes should equal to '
                  'number_of_byte')
            return

        self.noise_mean_el = params['noise_mean_el']
        self.noise_sigma_el = params['noise_sigma_el']
        self.no_time_samples = params['number_of_time_samples']
        self.constant = float(params['constant'])
        self.sp_curve = params['sp_curve']
        self.bytes_curve = params['bytes_curve']
        self.key = [i for i in range(params['number_of_bytes'])]
        self.seed = seed
        self.leakage_model_name = params['leakage_model_name']
        self.masking = params['masking']
        self.number_of_masking_bytes = params['num_of_masking_bytes']
        self.shuffle = params['shuffle']
        self.shift = params['shift']

        if self.leakage_model_name == "default":
            self.leakage_model = HammingPrecomputedModel()

        self.value_dtype = np.dtype([("plaintext", np.uint8, (self.number_of_bytes, 1)),
                                     ("leakage_model_output", np.uint8, (self.number_of_bytes, self.no_time_samples)),
                                     ("key", np.uint8, (self.number_of_bytes, 1)),
                                     ("power_components", np.float64, (3, self.no_time_samples)),
                                     ("mask", np.uint8, (self.number_of_bytes, self.number_of_masking_bytes, 1))])
        AbstractContainer.__init__(self, params['number_of_traces'], **kwargs)

    @staticmethod
    def generate_bytes_curve(idx_sbox):
        n = len(idx_sbox)
        curve = np.ones((n, 1), dtype=np.float64)
        if n <= 2:
            return curve
        else:
            for i in range(n // 2):
                if n % 2 == 0:
                    curve[i] /= 2 ** (n // 2 - i - 1)
                    curve[n - i - 1] /= 2 ** (n // 2 - i - 1)
                else:
                    curve[i] /= 2 ** (n // 2 - i)
                    curve[n - i - 1] /= 2 ** (n // 2 - i)
        return curve

    def generate_trace(self, idx):
        np.random.seed(seed=self.seed ^ idx)  # for reproducibility
        value = np.zeros((), dtype=self.value_dtype)
        # pre-generate ciphertext with binomial distribution
        value["leakage_model_output"] = np.random.binomial(n=8, p=0.5,
                                                           size=(self.number_of_bytes, self.number_of_time_samples))

        # we only encrypt attack_sample_point and 2 points after it (3 sample points in total)
        value["key"] = np.array(self.key, ndmin=2).T
        if self.attack_sample_point + 2 < self.number_of_time_samples:
            value["plaintext"] = np.random.randint(0, 256, (self.number_of_bytes, 1), np.uint8)
            cipher = value["plaintext"] ^ value["key"]
            if self.masking:
                value["mask"] = np.random.randint(0, 256,
                                                  (self.number_of_bytes, self.number_of_masking_bytes, 1), np.uint8)
                r_bytes = np.zeros((self.number_of_bytes, 1))
            for i in range(self.number_of_bytes):
                sbox_output = sbox[cipher[i]]
                if self.masking:
                    for j in range(self.number_of_masking_bytes):
                        sbox_output ^= value["mask"][i][j]
                        r_bytes[i] += self.leakage_model(value["mask"][i][j])

                cipher[i] = self.leakage_model(sbox_output)
            # keep the same value along the time axis
            cipher = cipher.repeat(3, axis=1)
            if self.masking:
                r_bytes = r_bytes.repeat(self.no_time_samples, axis=1)
        else:
            print('[INFO] attack sample point is too late, pls choose earlier ones')
            return
        value["leakage_model_output"][:, self.attack_sample_point:self.attack_sample_point + 3] = cipher
        if self.masking:
            value["leakage_model_output"] = np.add(value["leakage_model_output"], r_bytes)

        # generate electronic noise
        mean_el = np.array([self.noise_mean_el] * self.number_of_time_samples)
        cov_el = np.diag([self.noise_sigma_el] * self.number_of_time_samples)
        p_el = np.random.multivariate_normal(mean_el, cov_el).T  # 1 * no_time_samples
        p_el[self.attack_sample_point + 1] = p_el[self.attack_sample_point]
        # p_el.astype(np.float64)

        # generate curve of bytes
        curve_exp, curve_switch = 1.0, 1.0
        if self.bytes_curve:
            curve_exp = self.generate_bytes_curve(self.idx_exp)
            curve_switch = self.generate_bytes_curve(self.idx_switch)

        # coefficients of exp (a) and switch (b)
        # a[i] = idx_exp * curve_exp[i]; b[i] == idx_switch * curve_switch[i]
        p_exp = self.linear_coefficient_exp * np.sum(value["leakage_model_output"][self.idx_exp] * curve_exp, axis=0)
        p_switch = self.linear_coefficient_switch * np.sum(value["leakage_model_output"][self.idx_switch] * curve_switch, axis=0)
        for i in range(self.attack_sample_point, self.attack_sample_point + 3):
            p_el[i] = p_el[i] * self.sp_curve[i - self.attack_sample_point]
            p_exp[i] = p_exp[i] * self.sp_curve[i - self.attack_sample_point]
            p_switch[i] = p_switch[i] * self.sp_curve[i - self.attack_sample_point]
        power = p_exp + p_switch + p_el + self.constant

        # 3 * no_time_samples, used in calculation of real snr
        value["power_components"] = np.vstack((np.vstack((p_el, p_exp)), p_switch))

        if self.shuffle:
            np.random.shuffle(np.transpose(power))

        if self.shift:
            shift_step = np.random.randint(0, self.no_time_samples - (self.attack_sample_point + 2))
            power = np.roll(power, shift_step)
            power[:shift_step] = 0

        return Trace(power, value)

    def calc_snr(self, type='theo'):
        """
        calculation of snr: var(p_exp) / var(p_switch + p_el)
        1. theoretical snr
        1.1 assumptions:
        (a) p_switch and p_el are independent (cov(p_switch, p_el) = 0);
        (b) s-boxes are independent with each other, therefore var(sum(a_i * p_exp_i)) = sum((a_i ^ 2) * var(p_exp_i))
            and p_switch is the same;
        (c) p_exp and p_switch both are binomial distributions in HW model (0-7 after processed by the leakage model),
            therefore var(p_exp_i) = 8 * p * (1-p) = var(p_switch_i), where p=0.5
        1.2 formula
        snr = var(p_exp) / (var(p_switch) + var(p_el))
        2. real snr
        we take advantages of engines of lascar to calculate real var(p_exp), var(p_switch) and var(p_el) after we
        create instances of this class
        """
        # theoretical snr
        if type == 'theo':
            var_p_exp = 0.25 * 8 * len(self.idx_exp)
            var_p_switch = 0.25 * 8 * len(self.idx_switch)
            if not ((self.linear_coefficient_switch ** 2) * var_p_switch + self.noise_sigma_el):
                raise ValueError(
                    "divided by 0 occurs, cannot calculate theoretical snr, pls change parameters"
                )
            else:
                snr_theo = self.linear_coefficient_exp ** 2 * var_p_exp / ((self.linear_coefficient_switch ** 2) *
                                                                           var_p_switch + self.noise_sigma_el)
                snr_theo = [snr_theo] * self.no_time_samples
                return snr_theo
        if type == 'real':
            # real snr
            p_exp_m = np.zeros((self.number_of_traces, self.no_time_samples))
            p_switch_m = np.zeros((self.number_of_traces, self.no_time_samples))
            p_el_m = np.zeros((self.number_of_traces, self.no_time_samples))
            for i in range(self.number_of_traces):
                p_el_m[i, :] = self[i].value["power_components"][0, :]
                p_exp_m[i, :] = self[i].value["power_components"][1, :]
                p_switch_m[i, :] = self[i].value["power_components"][2, :]
            var_exp = np.var(p_exp_m, axis=0)
            var_el_switch = np.var(p_el_m + p_switch_m, axis=0)
            snr_real = var_exp / var_el_switch
            return snr_real

    def plot_traces(self, idx_traces):
        if not isinstance(idx_traces, list):
            print('[INFO] idx_traces should be a list')
        import matplotlib.pyplot as plt
        plt.title('simulated traces')
        plt.xlabel('time')
        plt.xticks(np.arange(0, self.number_of_time_samples, 1))
        plt.ylabel('power')
        for i in tqdm(range(idx_traces)):
            plt.plot(self[i].leakage)
        plt.show()


class SimulatedPowerTraceFixedRandomContainer(AbstractContainer):

    def __init__(self, number_of_traces=None, config_file_name='fixed_random_traces.yaml', seed=1337, **kwargs):
        params = TraceConfig().get_config(config_file_name)
        if number_of_traces:
            self.number_of_trace = number_of_traces
        else:
            self.number_of_traces = params['number_of_traces']
        self.fixed_set = params['fixed_set']
        self.number_of_bytes = params['number_of_bytes']
        self.number_of_time_samples = params['number_of_time_samples']
        self.attack_sample_point = params['attack_sample_point']
        self.linear_coefficient_exp = float(params['linear_coefficient_exp'])
        self.idx_exp = params['idx_exploitable_bytes']
        if not len(self.idx_exp) == self.number_of_bytes:
            print('[INFO] total number of exploitable signal bytes and switching noise bytes should equal to '
                  'number_of_byte')
            return

        self.noise_sigma_el = params['noise_sigma_el']
        self.noise_mean_el = params['noise_mean_el']
        self.no_time_samples = params['number_of_time_samples']
        self.constant = float(params['constant'])
        self.sp_curve = params['sp_curve']
        self.bytes_curve = params['bytes_curve']
        self.key = [i for i in range(params['number_of_bytes'])]
        self.seed = seed
        self.leakage_model_name = params['leakage_model_name']
        self.masking = params['masking']
        self.number_of_masking_bytes = params['num_of_masking_bytes']
        self.shuffle = params['shuffle']
        self.shift = params['shift']

        if self.leakage_model_name == "default":
            self.leakage_model = HammingPrecomputedModel()

        self.value_dtype = np.dtype([("plaintext", np.uint8, (self.number_of_bytes, 1)),
                                     ("leakage_model_output", np.uint8, (self.number_of_bytes, self.no_time_samples)),
                                     ("key", np.uint8, (self.number_of_bytes, 1)),
                                     ("power_components", np.float64, (2, self.no_time_samples)),
                                     ("mask", np.uint8, (self.number_of_bytes, self.number_of_masking_bytes, 1)),
                                     ("trace_idx", np.uint8, ())])
        AbstractContainer.__init__(self, params['number_of_traces'], **kwargs)

    @staticmethod
    def generate_bytes_curve(idx_sbox):
        n = len(idx_sbox)
        curve = np.ones((n, 1), dtype=np.float64)
        if n <= 2:
            return curve
        else:
            for i in range(n // 2):
                if n % 2 == 0:
                    curve[i] /= 2 ** (n // 2 - i - 1)
                    curve[n - i - 1] /= 2 ** (n // 2 - i - 1)
                else:
                    curve[i] /= 2 ** (n // 2 - i)
                    curve[n - i - 1] /= 2 ** (n // 2 - i)
        return curve

    def generate_trace(self, idx):
        value = np.zeros((), dtype=self.value_dtype)
        value["trace_idx"] = idx
        value["key"] = np.array(self.key, ndmin=2).T
        # random set generator
        np.random.seed(seed=self.seed ^ idx)  # for reproducibility
        value["leakage_model_output"] = np.random.binomial(n=8, p=0.5,
                                                           size=(self.number_of_bytes, self.number_of_time_samples))
        if idx % 2 == 0:
            value["plaintext"] = np.random.randint(0, 256, (self.number_of_bytes, 1), np.uint8)
        else:
            # fixed set generator
            if not len(self.fixed_set) == self.number_of_bytes:
                print('[INFO] the size of fixed set is conflicted with the number of bytes')
                return
            else:
                value["plaintext"] = np.array(self.fixed_set, ndmin=2).T

        if self.attack_sample_point + 2 < self.number_of_time_samples:
            value["plaintext"] = np.random.randint(0, 256, (self.number_of_bytes, 1), np.uint8)
            cipher = value["plaintext"] ^ value["key"]
            if self.masking:
                value["mask"] = np.random.randint(0, 256,
                                                  (self.number_of_bytes, self.number_of_masking_bytes, 1), np.uint8)
                r_bytes = np.zeros((self.number_of_bytes, 1))
            for i in range(self.number_of_bytes):
                sbox_output = sbox[cipher[i]]
                if self.masking:
                    for j in range(self.number_of_masking_bytes):
                        sbox_output ^= value["mask"][i][j]
                cipher[i] = self.leakage_model(sbox_output)
                if self.masking:
                    r_bytes[i] += self.leakage_model(value["mask"][i][j])
            # # keep the same value along the time axis
            cipher = cipher.repeat(3, axis=1)
        else:
            print('[INFO] attack sample point is too late, pls choose earlier ones')
            return
        value["leakage_model_output"][:, self.attack_sample_point:self.attack_sample_point + 3] = cipher
        if self.masking:
            value["leakage_model_output"] = np.add(value["leakage_model_output"], r_bytes)

        # generate electronic noise
        mean_el = np.array([self.noise_mean_el] * self.number_of_time_samples)
        cov_el = np.diag([self.noise_sigma_el] * self.number_of_time_samples)
        p_el = np.random.multivariate_normal(mean_el, cov_el).T  # no_time_samples * 1
        p_el[self.attack_sample_point + 1] = p_el[self.attack_sample_point]
        # p_el.astype(np.float64)

        # generate curve of bytes
        curve_exp = 1.0
        if self.bytes_curve:
            curve_exp = self.generate_bytes_curve(self.idx_exp)

        # coefficients of exp (a) and switch (b)
        # a[i] = idx_exp * curve_exp[i]; b[i] == idx_switch * curve_switch[i]
        p_exp = self.linear_coefficient_exp * np.sum(value["leakage_model_output"][self.idx_exp] * curve_exp, axis=0)
        for i in range(self.attack_sample_point, self.attack_sample_point + 3):
            p_el[i] = p_el[i] * self.sp_curve[i]
            p_exp[i] = p_exp[i] * self.sp_curve[i]
        power = p_exp + p_el + self.constant

        # 3 * no_time_samples, used in calculation of real snr
        value["power_components"] = np.vstack((p_el, p_exp))

        if self.shuffle:
            np.random.shuffle(np.transpose(power))

        if self.shift:
            shift_step = np.random.randint(0, self.no_time_samples - (self.attack_sample_point + 2))
            power = np.roll(power, shift_step)
            power[:shift_step] = 0

        return Trace(power, value)

    def calc_snr(self, type='theo'):
        """
        calculation of snr: var(p_exp) / var(p_switch + p_el)
        1. theoretical snr
        1.1 assumptions:
        (a) p_switch and p_el are independent (cov(p_switch, p_el) = 0);
        (b) s-boxes are independent with each other, therefore var(sum(a_i * p_exp_i)) = sum((a_i ^ 2) * var(p_exp_i))
            and p_switch is the same;
        (c) p_exp and p_switch both are binomial distributions in HW model (0-7 after processed by the leakage model),
            therefore var(p_exp_i) = len(idx_exp) * 8 * p * (1-p); var(p_switch_i) = len(idx_switch) * p * (1-p), where p=0.5
        1.2 formula
        snr = var(p_exp) / (var(p_switch) + var(p_el))
        2. real snr
        we take advantages of engines of lascar to calculate real var(p_exp), var(p_switch) and var(p_el) after we
        create instances of this class
        """
        # theoretical snr
        if type == 'theo':
            var_p_exp = 0.25 * 8 * len(self.idx_exp)
            if not self.noise_sigma_el:
                raise ValueError(
                    "divided by 0 occurs, cannot calculate theoretical snr, pls change parameters"
                )
            else:
                snr_theo = self.linear_coefficient_exp ** 2 * var_p_exp / self.noise_sigma_el
                snr_theo = [snr_theo] * self.no_time_samples
                return snr_theo
        if type == 'real':
            # real snr
            p_exp_m = np.zeros((self.number_of_traces, self.no_time_samples))
            p_el_m = np.zeros((self.number_of_traces, self.no_time_samples))
            for i in range(self.number_of_traces):
                p_el_m[i, :] = self[i].value["power_components"][0, :]
                p_exp_m[i, :] = self[i].value["power_components"][1, :]
            var_exp = np.var(p_exp_m, axis=0)
            var_el_switch = np.var(p_el_m, axis=0)
            snr_real = var_exp / var_el_switch
            return snr_real

    def plot_traces(self, idx_traces):
        if not isinstance(idx_traces, list):
            print('[INFO] idx_traces should be a list')
        import matplotlib.pyplot as plt
        plt.figure(0)
        plt.title('simulated traces')
        plt.xlabel('time')
        plt.xticks(np.arange(0, self.number_of_time_samples, 1))
        plt.ylabel('power')
        for i in tqdm(idx_traces):
            plt.plot(self[i].leakage)
        plt.show()


if __name__ == '__main__':
    SPT = SimulatedPowerTraceContainer(50)
