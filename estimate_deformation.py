from mymodel import UNet_Att_Cascade6_Cube128_Regression
from torch.utils import data
from myutils import file_exist, join_path, ls, gn, get_nifti_pixdim, load_nifti, \
    load_nifti_simple, save_nifti, save_nifti_simple, try_load_nifti, \
    barycentric_coordinate, center_crop, z_score, Timer, minibar, mkdir, gd, masked_mean, masked_std
import numpy as np
from scipy.ndimage import zoom as zoom_image
import argparse
from torch.optim import Adam as AdamOptimizer
from model_trainer import ModelTrainer_PyTorch
import torch
import torch.nn as nn

dataloader_numworkers = 12
model_in_channels, model_out_channels = 1, 1
initial_lr = 0.0002
num_epochs = 100
only_run_inference = True
load_model_path = "/hd/tumor_analysis/digicare/experiments/022_ki67_real_tumor/archived_models/attunet_2mm_600samps_withneg_nocarve_T1only.model"
inference_output_dir = mkdir('Preprocessed_data/Estimated_deform/')
tiantan_ki67_samples = ['xlsx/ki67_train.xlsx', 'xlsx/ki67_val.xlsx']

data_samples = {}

parser = argparse.ArgumentParser(
    description='Run ablation study for tumor diagnosis.')
parser.add_argument(
    '--run-on-which-gpu', '-g', type=int, 
    help='GPU ID (starts with 0).', 
    required=False,
    default=0)
args = vars(parser.parse_args())

# step 3: train model
gpu = int(args['run_on_which_gpu'])
model = UNet_Att_Cascade6_Cube128_Regression(model_in_channels, model_out_channels).to('cuda:%d' % gpu)
optim = AdamOptimizer(model.parameters(), lr=initial_lr, betas=(0.9, 0.999))

class Trainer(ModelTrainer_PyTorch):
    def __init__(self, 
        output_folder: str, 
        gpu_index: int, 
        model, optim, 
        train_loader, val_loader):
        super().__init__(output_folder=output_folder, 
            gpu_index=gpu_index, model=model, optim=optim, lr_scheduler=None, 
            train_loader=train_loader, val_loader=val_loader, test_loader=None)        
    def _on_epoch(self, epoch: int, msg: str, data_loader, phase):
        self.model.train() if phase=='train' else self.model.eval()
        timer = Timer()
        losses = []
        for batch_idx, (sample) in enumerate(data_loader):
            _, input_data, output_disp_data = sample
            input_data, output_disp_data = input_data.to('cuda:%d' % gpu).float(), output_disp_data.to('cuda:%d' % gpu).float()
            loss_function = nn.MSELoss()
            if phase=='train':
                output_estimation = self.model(input_data)
                assert output_estimation.shape == output_disp_data.shape
                loss = loss_function(output_estimation, output_disp_data)
                optim.zero_grad()
                loss.backward()
                optim.step()
            else:
                with torch.no_grad():
                    output_estimation = self.model(input_data)
                    assert output_estimation.shape == output_disp_data.shape
                    loss = loss_function(output_estimation, output_disp_data)
            losses.append(loss.item())
            minibar(msg, batch_idx+1, len(data_loader), timer.elapsed(), last='%.4f' % loss.item())
        return np.mean(losses)

trainer = Trainer('.', gpu, model, optim, None, None)
trainer.train(num_epochs=0) # do not train
trainer.load_model(load_model_path)

def run_inference_on_data(data_sample, model, output_dir):
    case_name, input_image, input_mask = data_sample
    save_path = join_path(output_dir, '%s_estimated_deform.nii.gz' % case_name)
    if try_load_nifti(save_path) == True:
        return save_path
    print("case name:", case_name)
    print("input:", input_image)
    # during training the data samples are resampled to 2mm
    input_data, input_header = load_nifti(input_image)
    input_mask = load_nifti_simple(input_mask)
    input_data = z_score(input_data, input_mask > 0.5)
    input_128cube = zoom_image(input_data, 0.5)
    assert input_128cube.shape[0] == input_128cube.shape[1] == input_128cube.shape[2] == 128
    # save_nifti_simple(input_128cube, join_path(output_dir, '%s_in.nii.gz' % case_name))
    input_128cube = input_128cube[None, None]
    with torch.no_grad():
        model.eval()
        input_128cube = torch.tensor(input_128cube).to('cuda:%d' % gpu).float()
        output_128cube = model(input_128cube)
        output_128cube = output_128cube[0,0].detach().float().cpu().numpy()
    # resample output
    output_256cube = zoom_image(output_128cube, 2).astype('float32')
    save_nifti_simple(output_256cube, save_path)
    print("output:", save_path)
    return save_path

# test on tiantan data
from database import Database
def collect_tiantan_ki67_data():
    data_samples = {}
    all_ki67_cases = []
    for xlsx in tiantan_ki67_samples:
        xlsxobj = Database(db_keys=['subject_name'], xlsx_file=xlsx)
        all_ki67_cases += xlsxobj.data_dict['subject_name']

    for patient_name in all_ki67_cases:
        patient_folder = join_path('/hd/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/Preprocessed_data/', patient_name)
        in_image = ''
        if file_exist(join_path(patient_folder, 'origres', 't1.nii.gz')):
            in_image = join_path(patient_folder, 'origres', 't1.nii.gz')
        else:
            raise RuntimeError('cannot find T1 image. %s' % in_image)
        in_mask = join_path(patient_folder, 'origres', 'brain_mask.nii.gz')
        assert file_exist(in_image), in_image
        assert file_exist(in_mask), in_mask
        data_samples[patient_name] = {
            'input': in_image,
            'mask': in_mask,
        }
    return data_samples

all_test_samples = collect_tiantan_ki67_data()
print(all_test_samples)

from data_processing import create_database_for_tumor_diagnosis
input_xlsx = '/hd/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/xlsx/processed/tiantan_ljj_v1.0.xlsx'
output_xlsx = '/hd/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/xlsx/processed/tiantan_ljj_v1.0_for_ki67.xlsx'

database = create_database_for_tumor_diagnosis(input_xlsx)

for i, case in enumerate(all_test_samples):
    print([i, len(all_test_samples)], case)
    estimated_ki67_deform_save_path = run_inference_on_data((case, all_test_samples[case]['input'], all_test_samples[case]['mask']), trainer.model, inference_output_dir)
    # 将生成的ki67形变数据补写到原有xlsx表格中
    ind, record = database.get_record_from_key_val_pair('subject_name', case)
    assert record is not None, 'cannot find case "%s" in database.' % case
    record['ki67_deform'] = estimated_ki67_deform_save_path
    database.set_record(ind, record)

database.export_xlsx(output_xlsx)

