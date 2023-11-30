"""
 Extract audio features of mel-spectrogram and gcc-phat to build the AFPILD dataset

 samples were generated with the four channels of waveforms as input,
 ideally, each sample was designed to contain a single footstep event.

Author: Shichao WU

"""

import librosa
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# dataset path
ori_data_dir = "../data/AFPILD_v1"

# generated feat path
afpild_fea_dir = '../data/audio_feature'

# manually create two empty directory at the start for feature saving
gcc_cachedir = "gcc"
spec_cachedir = "spec"

if not os.path.exists(os.path.join(afpild_fea_dir, gcc_cachedir)):
    os.makedirs(os.path.join(afpild_fea_dir, spec_cachedir))
    os.makedirs(os.path.join(afpild_fea_dir, gcc_cachedir))

cloth_train_df = pd.DataFrame(
    columns=["fea_spec", "fea_gcc", "loc_azimuth", "loc_x", "loc_y", "subject_label"])
cloth_test_df = pd.DataFrame(columns=["fea_spec", "fea_gcc", "loc_azimuth", "loc_x", "loc_y", "subject_label"])

shoe_train_df = pd.DataFrame(columns=["fea_spec", "fea_gcc", "loc_azimuth", "loc_x", "loc_y", "subject_label"])
shoe_test_df = pd.DataFrame(columns=["fea_spec", "fea_gcc", "loc_azimuth", "loc_x", "loc_y", "subject_label"])

all_df = pd.DataFrame(columns=["fea_spec", "fea_gcc", "loc_azimuth", "loc_x", "loc_y", "subject_label"])

# sub1, sub2, ...., sub40.
ori_sub_dir = sorted(os.listdir(ori_data_dir))

sample_num = 1
for i in range(len(ori_sub_dir)):
    sub_dir = os.path.join(ori_data_dir, ori_sub_dir[i])
    if os.path.isfile(sub_dir):
        continue

    audio_dir = sorted(os.listdir(sub_dir))
    print("======== subject: {} ========".format(ori_sub_dir[i][1:]))
    # iterate over each recorded audio file
    for j in range(1, 5):
        print("session: {}".format(j))

        audio_fil_name = 's' + ori_sub_dir[i][1:] + '_' + str(j) + '_' + 'footstep_audio.wav'
        meta_fil_name = 's' + ori_sub_dir[i][1:] + '_' + str(j) + '_' + 'footstep_annotation.csv'

        # load the footstep events separation sampling points
        meta_fil = pd.read_csv(os.path.join(sub_dir, meta_fil_name))
        # load audio
        input_audio, sr = librosa.load(os.path.join(sub_dir, audio_fil_name), sr=16000, mono=False)

        # select the corresponding FEs separation sampling points
        # j starts from 5 to skip the silence beginning
        for sample_cnt in range(5, len(meta_fil) - 3):
            # for a single footstep event
            sample_audio = input_audio[:, meta_fil['sample_loc'][sample_cnt]: meta_fil['sample_loc'][sample_cnt + 1]]

            # # visualization
            # plt.plot(sample_audio[0, :])
            # plt.plot(sample_audio[1, :]+0.2)
            # plt.plot(sample_audio[2, :] + 0.2*2)
            # plt.plot(sample_audio[3, :] + 0.2*3)
            # plt.show()

            # window length belongs to 20-40 ms, here we choose 20ms, with an overlapping length of 10ms.
            # for a single footstep event
            win_len = int(0.02 * sr)
            step_len = int(0.01 * sr)

            # for padding
            fixed_sample_len = int(0.64 * sr)
            fixed_pad_len = int(fixed_sample_len / step_len)

            # a. mel-spectrogram feature extraction  <<<====================
            spectra_fea = []
            linear_spectra = []
            for ch_cnt in range(4):
                spec_ch = librosa.feature.melspectrogram(y=sample_audio[ch_cnt, :], sr=sr, n_fft=2048,
                                                         hop_length=step_len, n_mels=64, win_length=win_len,
                                                         window='hann', fmin=0, fmax=8000)
                spec_ch = np.log(spec_ch)

                # z-score normalization
                std_val = spec_ch.std()
                mean_val = spec_ch.mean()
                spec_ch = (spec_ch - mean_val) / std_val + 1e-8

                # padding the spectrogram to generate a fixed shape of 64 x 64
                f_len = spec_ch.shape[0]
                spec_ch_padded = np.zeros((f_len, fixed_pad_len), dtype='float32')
                tmp = spec_ch[:, :fixed_pad_len]
                spec_ch_padded[:, 0:tmp.shape[1]] = tmp  # ==> for saving

                spectra_fea.append(spec_ch_padded)

                # linear spectrogram extraction
                linear_spectra_ch = librosa.core.stft(np.asfortranarray(sample_audio[ch_cnt, :]), n_fft=2048,
                                                      hop_length=step_len, win_length=win_len, window='hann')
                linear_spectra.append(linear_spectra_ch)

            linear_spectra = np.array(linear_spectra).T  # (time_dim x freq_dim x channel_num)

            # b. gcc-phat feature extraction  <<<====================
            gcc_channels = 6
            gcc_fea = np.zeros((fixed_pad_len, 64, gcc_channels))  # (time_dim x freq_dim x channel_num)
            cnt = 0
            for m in range(linear_spectra.shape[-1]):
                for n in range(m + 1, linear_spectra.shape[-1]):
                    R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                    cc = np.fft.irfft(np.exp(1.j * np.angle(R)))
                    cc = np.concatenate((cc[:, -64 // 2:], cc[:, :64 // 2]), axis=-1)

                    # z-score normalization
                    std_val = cc.std()
                    mean_val = cc.mean()
                    cc = (cc - mean_val) / std_val + 1e-8

                    # padding to the same length of 64 x 64
                    tmp = cc[:fixed_pad_len, :]
                    gcc_fea[0:tmp.shape[0], :, cnt] = tmp
                    cnt += 1

            gcc_fea = gcc_fea.transpose((1, 0, 2))  # (freq_dim x time_dim x channel_num)
            spectra_fea = np.array(spectra_fea).transpose((1, 2, 0))  # (freq_dim x time_dim x channel_num)

            # feature saving  <<<====================
            fname_spec = os.path.join(spec_cachedir, f"afpild_fe1_{audio_fil_name[:5]}_melspec_{sample_num}.npy")
            fname_gcc = os.path.join(gcc_cachedir, f"afpild_fe1_{audio_fil_name[:5]}_gccphat_{sample_num}.npy")
            sample_num += 1

            # saving
            # np.save(os.path.join(self.spectrogram_cachedir, item['filename_audio']).replace(".wav", ".npy"), spec)
            np.save(os.path.join(afpild_fea_dir, fname_spec), spectra_fea)
            np.save(os.path.join(afpild_fea_dir, fname_gcc), gcc_fea)

            session = int(audio_fil_name[4])

            # convert cartesian to polar
            x, y = meta_fil['loc_x'][sample_cnt], meta_fil['loc_y'][sample_cnt]
            azimuth = np.arctan2(y, x) * 180 / np.pi  # in degree,  azimuth in [-180, 180]

            df_new_row = pd.DataFrame({"fea_spec": [fname_spec], "fea_gcc": [fname_gcc], "loc_azimuth": [azimuth],
                                       "loc_x": [meta_fil['loc_x'][sample_cnt]], "loc_y": [meta_fil['loc_y'][sample_cnt]],
                                       "subject_label": [f"S{audio_fil_name[1:3]}"]})
            if session == 1:
                cloth_train_df = pd.concat([cloth_train_df, df_new_row], ignore_index=True)
                shoe_train_df = pd.concat([shoe_train_df, df_new_row], ignore_index=True)
                all_df = pd.concat([all_df, df_new_row], ignore_index=True)

            elif session == 2:
                cloth_test_df = pd.concat([cloth_test_df, df_new_row], ignore_index=True)
                shoe_train_df = pd.concat([shoe_train_df, df_new_row], ignore_index=True)
                all_df = pd.concat([all_df, df_new_row], ignore_index=True)

            elif session == 3:
                cloth_test_df = pd.concat([cloth_test_df, df_new_row], ignore_index=True)
                shoe_test_df = pd.concat([shoe_test_df, df_new_row], ignore_index=True)
                all_df = pd.concat([all_df, df_new_row], ignore_index=True)

            elif session == 4:
                cloth_train_df = pd.concat([cloth_train_df, df_new_row], ignore_index=True)
                shoe_test_df = pd.concat([shoe_test_df, df_new_row], ignore_index=True)
                all_df = pd.concat([all_df, df_new_row], ignore_index=True)

            else:
                print('========>>>>>>>>>>  ERROR !!!! <<<<<<<<<<============')

# saving meta info into .csv file
data_len = len(all_df)
samp_idx = np.random.permutation(np.arange(data_len))
train_len = int(data_len * 0.5)

train_pd_rd = all_df.iloc[samp_idx[:train_len]]
train_pd_rd.reset_index(drop=True, inplace=True)

test_pd_rd = all_df.iloc[samp_idx[train_len:]]
test_pd_rd.reset_index(drop=True, inplace=True)

train_pd_rd.to_csv(os.path.join(afpild_fea_dir, "AFPILD_FE1_rd_train.csv"))
test_pd_rd.to_csv(os.path.join(afpild_fea_dir, "AFPILD_FE1_rd_test.csv"))
# all_df.to_csv(os.path.join(afpild_fea_dir, "AFPILD_FE1_all.csv"))

cloth_train_df.to_csv(os.path.join(afpild_fea_dir, 'AFPILD_FE1_cloth_train.csv'))
cloth_test_df.to_csv(os.path.join(afpild_fea_dir, 'AFPILD_FE1_cloth_test.csv'))

shoe_train_df.to_csv(os.path.join(afpild_fea_dir, 'AFPILD_FE1_shoe_train.csv'))
shoe_test_df.to_csv(os.path.join(afpild_fea_dir, 'AFPILD_FE1_shoe_test.csv'))

print("Finished to create the AFPILD_FE1 dataset with a SINGLE footstep event to form ONE sample!")
