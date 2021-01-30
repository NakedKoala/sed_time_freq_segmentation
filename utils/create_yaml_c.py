from config_c import config
from pathlib import Path 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
import yaml


def folder_to_df(path, chest_loc='*', ac_mode='*', rc_equip='AKGC417L'):

    #chest_loc: Tc, A, P, L, l, r
    #ac_mode: sc, mc
    #rc_equip: AKGC417, other stuff

    # 162_2b4_Tc_mc_AKGC417L.wav

    fnames = list(path.glob(f'*_{chest_loc}_{ac_mode}_{rc_equip}.wav'))

    data_dict = {
        'patient': [],
        'recording_index': [],
        'chest_loc': [],
        'ac_mode': [],
        'rc_equip': [],
        'file_path':[]
    }

    for fname in fnames:
       patient, recording_index, chest_loc, ac_mode, rc_equip = str(fname).split("/")[-1].split(".")[0].split("_")
       data_dict['patient'].append(patient)
       data_dict['recording_index'].append(recording_index)
       data_dict['chest_loc'].append(chest_loc)
       data_dict['ac_mode'].append(ac_mode)
       data_dict['rc_equip'].append(rc_equip)
       data_dict['file_path'].append(str(fname))

    return pd.DataFrame(data_dict)



def generate_train_folds(config):
    df = folder_to_df(config.base_dir, chest_loc='*', ac_mode='*', rc_equip='AKGC417L')
    df['fold'] = -1
    kf = KFold(n_splits=config.num_folds, random_state=config.seed, shuffle=True)
    for fold, (trn_index, val_index) in enumerate(kf.split(df['file_path'])):
        df.loc[val_index, "fold"] = fold
    return df 

def record_to_data(config, record):

  data = {}
  data['fold'] = record['fold']
  data['fname'] = str(record['file_path']).split('/')[-1]
  data['events'] = []

  label_path = str(record['file_path']).split(".")[-2] + '.txt'
  label_df = pd.read_csv(label_path, sep='\t', names=['tmin', 'tmax', 'crackles', 'wheezes'])

  for index, row in label_df.iterrows():
    if int(row['crackles']) == 1:
       data['events'].append({ 
                                'event_audio_name': data['fname'],
                                'onset': row['tmin'].item(),
                                'offset': row['tmax'].item(),
                                'event_label': 'crackles'
                              })
    if int(row['wheezes']) == 1:
      data['events'].append({ 
                            'event_audio_name': data['fname'],
                            'onset': row['tmin'].item(),
                            'offset': row['tmax'].item(),
                            'event_label': 'wheezes'
                          })
  return data 

def create_yaml(config):

   cv_fold = generate_train_folds(config)
   data_list = []
   for fold in range(config.num_folds):
       print(f'create yaml for fold {fold} ... ')
       fold_df = cv_fold[cv_fold['fold'] == fold]

       for index, record in fold_df.iterrows():

           data_list.append(record_to_data(config, record))

   with open(config.out_yaml_path, 'w') as f:
        yaml.dump(data_list, f, default_flow_style=False)
        
   print('Write out data yaml to {}'.format(config.out_yaml_path))

if __name__ == '__main__':
    create_yaml(config)


      



