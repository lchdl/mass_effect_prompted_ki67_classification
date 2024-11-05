import numpy as np
from myutils import load_nifti_simple, save_nifti_simple, get_nifti_dtype, \
    abs_path, gd, rm, join_path, mv, file_exist, mkdir, run_shell, antsApplyTransforms
import argparse

def antsRegistrationCustom(source, target, warped, 
    interpolation_method='Linear', use_histogram_matching=False,
    dtype_check=False) -> str:

    assert interpolation_method in ['Linear', 'NearestNeighbor'], 'unknown interpolation method.'
    assert use_histogram_matching in [True, False], 'invalid parameter setting for "use_histogram_matching".'

    assert file_exist(source), 'Cannot open source image "%s". File not exist or insufficient privilege.' % source
    assert file_exist(target), 'Cannot open target image "%s". File not exist or insufficient privilege.' % target
    if dtype_check:
        if get_nifti_dtype(source) != get_nifti_dtype(target):
            raise RuntimeError('Source data type (%s) and target data type (%s) are different! '
                            'antsRegistration may fail if the two images have different storage data type.\n'
                            'Source image: "%s".\nTarget image: "%s".\n' % \
                                (get_nifti_dtype(source), get_nifti_dtype(target), source, target))
    
    output_directory = gd(abs_path(warped))
    mkdir(output_directory)
    output_transform_prefix = join_path(output_directory,'warp_')

    # fill in default configurations
    config = {
        'SyN_gradientStep' : 0.3, # 0.1 -> 0.3
        'SyN_updateFieldVarianceInVoxelSpace' : 3.0, # 3->2
        'SyN_totalFieldVarianceInVoxelSpace' : 0.0,
        'SyN_CC_neighborVoxels': 4,
        'SyN_convergence' : '100x70x50x20',
        'SyN_shrinkFactors' : '8x4x2x1',
        'SyN_smoothingSigmas' : '3x2x1x0',
        'collapse_output_transforms': 1, # 0 to disable
    }

    # generate registration command
    command = 'antsRegistration '
    command += '--verbose '
    command += '--dimensionality 3 '                                     # 3D image
    command += '--float 1 '                                              # 0: use float64, 1: use float32 (save mem)
    command += '--collapse-output-transforms %d ' % config['collapse_output_transforms']
    command += '--output [%s,%s] ' % (output_transform_prefix,warped)
    command += '--interpolation %s ' % interpolation_method
    command += '--use-histogram-matching %s ' % ( '0' if use_histogram_matching == False else '1')
    command += '--winsorize-image-intensities [0.005,0.995] '
    command += '--initial-moving-transform [%s,%s,1] ' % (target,source) # initial moving transform

    command += '--transform Rigid[0.1] '                                 # rigid transform
    command += '--metric MI[%s,%s,1,32,Regular,0.25] ' % (target,source)
    command += '--convergence [1000x500x250x0,1e-6,10] '
    command += '--shrink-factors 8x4x2x1 '
    command += '--smoothing-sigmas 3x2x1x0vox '

    command += '--transform Affine[0.1] ' # affine transform
    command += '--metric MI[%s,%s,1,32,Regular,0.25] ' % (target,source)
    command += '--convergence [1000x500x250x0,1e-6,10] '
    command += '--shrink-factors 8x4x2x1 '
    command += '--smoothing-sigmas 3x2x1x0vox '

    command += '--transform SyN[%f,%f,%f] ' % \
        (config['SyN_gradientStep'], config['SyN_updateFieldVarianceInVoxelSpace'], \
        config['SyN_totalFieldVarianceInVoxelSpace'])
    command += '--metric CC[%s,%s,1,%d] ' % (target,source, config['SyN_CC_neighborVoxels'])
    command += '--convergence [%s,1e-6,20] ' % (config['SyN_convergence'])
    command += '--shrink-factors %s ' % config['SyN_shrinkFactors']
    command += '--smoothing-sigmas %svox ' % config['SyN_smoothingSigmas']

    return command


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image', '-i', type=str, help='input image path. (the image to be deformed)', required=True)
    parser.add_argument('--output-image', '-o', type=str, help='output image path. (where the deformed image will be saved)', required=True)
    parser.add_argument('--output-displacement', '-d', type=str, help='output displacement image path.'\
                        '(where the deformation field will be saved)', required=True)
    parser.add_argument('--input-label', '-q', type=str, help='input label path.', required=True)
    parser.add_argument('--output-label', '-l', type=str, help='output label path.', required=True)
    parser.add_argument('--moving-image', '-m', type=str, help='moving image. '\
                        '(moving and fixed image will implicitly define a deformation field)', required=True)
    parser.add_argument('--fixed-image', '-f', type=str, help='fixed image.', required=True)
    args = vars(parser.parse_args())

    input_image = args['input_image']
    output_image = args['output_image']
    input_label = args['input_label']
    output_label = args['output_label']

    moving = join_path(args['moving_image'])
    fixed = join_path(args['fixed_image'])

    moved = '_moved.nii.gz'
    run_shell(antsRegistrationCustom(moving, fixed, moved))

    input_nohdr = 'input_nohdr.nii.gz'
    save_nifti_simple(load_nifti_simple(input_image), input_nohdr)

    affine = 'warp_0GenericAffine.mat'
    elastic = 'warp_1Warp.nii.gz'
    temp = '_temp.nii.gz'
    run_shell(antsApplyTransforms(input_nohdr, fixed, elastic, output_image))
    def calculate_displacement_from_deformation(field):
        D = load_nifti_simple(field)
        D = np.reshape(D, [*D.shape[:3], 3])
        D = np.transpose(D, [3,0,1,2])
        Dx, Dy, Dz = D[0], D[1], D[2]
        return np.sqrt(np.power(Dx, 2) + np.power(Dy, 2) + np.power(Dz, 2))
    save_nifti_simple(
        calculate_displacement_from_deformation(elastic),
        abs_path(args['output_displacement']))

    save_nifti_simple(load_nifti_simple(input_label), 'label_in.nii.gz')
    run_shell(antsApplyTransforms('label_in.nii.gz', fixed, elastic, 'label_out.nii.gz'))
    mv('label_out.nii.gz', output_label)

    print('file saved to "%s".' % abs_path(args['output_displacement']))
    
    #rm(moved)
    rm(input_nohdr)
    rm(affine)
    rm(elastic)
    rm(temp)
    rm('warp_1InverseWarp.nii.gz')

