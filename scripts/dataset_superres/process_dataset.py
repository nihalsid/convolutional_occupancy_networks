from pathlib import Path
import struct
import marching_cubes as mc
import numpy as np
import torch
from tqdm import tqdm
import traceback
import trimesh
import os

def visualize_point_list(grid, colors, output_path):
    f = open(output_path, "w")
    for i in range(grid.shape[0]):
        x, y, z = grid[i, 0], grid[i, 1], grid[i, 2]
        if colors is None:
            c = [1, 1, 1]
        else:
            c = [colors[i, 0], colors[i, 1], colors[i, 2]]
        f.write('v %f %f %f %f %f %f\n' % (x + 0.5, y + 0.5, z + 0.5, c[0], c[1], c[2]))
    f.close()


def visualize_sdf_color(sdf, color, output_path, level=0.75):
    vertices, triangles = mc.marching_cubes_color(sdf, color, level)
    mc.export_obj(vertices, triangles, output_path)


def to_point_list(s):
    return np.concatenate([c[:, np.newaxis] for c in np.where(s == True)], axis=1)


def visualize_probs(prob, output_path):
    prob = prob > 0.2
    point_list = to_point_list(prob)
    if point_list.shape[0] > 0:
        base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
        base_mesh.export(output_path)


def create_data_point_for_sdf(base_path, dataset_name, sample_name, output_folder):
    in_sdf = np.load(Path(base_path, "sdf_008", dataset_name, sample_name + ".npy"))
    tgt_sdf = np.load(Path(base_path, "sdf_064", dataset_name, sample_name + ".npy"))
    
    points_uniform_ratio = 1
    voxel_size_hr = 0.054167
    voxel_size_lr = 0.43334
    dim = 64
    points_sigma_0 = 0.01
    points_sigma_1 = 0.001
    points_sigma_2 = 0.1
    points_sigma_3 = 0.05
    points_sigma_4 = 0.005

    in_sdf[in_sdf > 3 * voxel_size_lr] = 3 * voxel_size_lr
    in_sdf[in_sdf < -3 * voxel_size_lr] = -3 * voxel_size_lr

    occupied_pts_surface_true = (np.stack(np.where(tgt_sdf <= 0.75 * voxel_size_hr)).T / dim) - 0.5
    points_vicinity = (np.stack(np.where((tgt_sdf <= 1 * voxel_size_hr) & (tgt_sdf > 0.75 * voxel_size_hr))).T / dim) - 0.5
    iou_points = (np.stack(np.where(tgt_sdf <= 1.5 * voxel_size_hr)).T / dim) - 0.5
    

    points_surface_sigma_0 = occupied_pts_surface_true + points_sigma_0 * np.random.randn(occupied_pts_surface_true.shape[0], 3)
    points_surface_sigma_1 = occupied_pts_surface_true + points_sigma_1 * np.random.randn(occupied_pts_surface_true.shape[0], 3)
    points_surface_sigma_2 = occupied_pts_surface_true + points_sigma_2 * np.random.randn(occupied_pts_surface_true.shape[0], 3)
    points_surface_sigma_3 = occupied_pts_surface_true + points_sigma_3 * np.random.randn(occupied_pts_surface_true.shape[0], 3)
    points_surface_sigma_4 = occupied_pts_surface_true + points_sigma_4 * np.random.randn(occupied_pts_surface_true.shape[0], 3)
    points_surface_true = occupied_pts_surface_true
    n_points_uniform = int((points_surface_true.shape[0] * 4)  * (points_uniform_ratio))
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = (points_uniform - 0.5)
    points = np.concatenate([points_surface_sigma_0, points_surface_sigma_1, points_surface_sigma_2, points_surface_sigma_3, points_surface_sigma_4, points_uniform, points_surface_true], axis=0)

    points_grid = ((points + 0.5) * dim).astype(np.uint32)
    points_grid_inside = ((points_grid >=0) & (points_grid < dim)).all(axis=1)
    occupancies = np.zeros((points_grid.shape[0]), dtype=np.bool)
    occupancies[points_grid_inside] = tgt_sdf[points_grid[points_grid_inside, 0], points_grid[points_grid_inside, 1], points_grid[points_grid_inside, 2]] <= 0.75 * voxel_size_hr

    iou_points_grid = np.clip(((iou_points + 0.5) * dim), 0, dim - 1).astype(np.uint32)
    iou_occupancies = tgt_sdf[iou_points_grid[:, 0], iou_points_grid[:, 1], iou_points_grid[:, 2]] <= 0.75 * voxel_size_hr

    # visualize_point_list(iou_points[iou_occupancies==True, :], None, f"dump/{sample_name}_iou_pts_true.obj")
    # visualize_point_list(iou_points, None, f"dump/{sample_name}_iou_pts.obj")
    # visualize_point_list(points[occupancies==True, :], None, f"dump/{sample_name}_occ.obj")
    # visualize_point_list(points, None, f"dump/{sample_name}_samples.obj")
    # visualize_probs(in_sdf < 0.5 * voxel_size_lr, f"dump/{sample_name}_insdf.obj")

    dtype = np.float16
    points = points.astype(dtype)
    iou_points = iou_points.astype(dtype)
    # print(points.shape)
    if points.shape[0] > 0:
        (output_folder / sample_name).mkdir(exist_ok=True, parents=True)
        np.savez_compressed((output_folder / sample_name / "points.npz"), points=points, occupancies=occupancies)
        np.savez_compressed((output_folder / sample_name / "points_iou.npz"), points=iou_points, occupancies=iou_occupancies)
        np.savez_compressed((output_folder / sample_name / "tsdf.npz"), geometry=in_sdf)    


def create_uniform_samples_for_sdf(base_path, dataset_name, sample_name, output_folder):
    voxel_size_hr = 0.054167
    dim = 64
    n_points_uniform = 100000
    n_files = 10
    padding = 0.1
    points_uniform = np.random.rand(n_points_uniform * n_files, 3).astype(np.float32)
    points_uniform = (points_uniform - 0.5) * (1 + padding)
    occupancies = np.zeros(n_points_uniform * n_files).astype(bool)

    tgt_sdf = np.load(Path(base_path, "sdf_064", dataset_name, sample_name + ".npy"))
    points_grid = ((points_uniform + 0.5) * dim).astype(np.uint32)
    points_grid_inside = ((points_grid >=0) & (points_grid < dim)).all(axis=1)
    occupancies[points_grid_inside] = tgt_sdf[points_grid[points_grid_inside, 0], points_grid[points_grid_inside, 1], points_grid[points_grid_inside, 2]] <= 0.75 * voxel_size_hr
    # visualize_point_list(points_uniform[occupancies==True, :], None, f"dump/{sample_name}_occ.obj")
    # visualize_point_list(points_uniform, None, f"dump/{sample_name}_samples.obj")
    points_uniform = points_uniform.reshape(n_files, n_points_uniform, 3)
    occupancies = occupancies.reshape(n_files, n_points_uniform)
    points_uniform = points_uniform.astype(np.float16)
    for file_idx in range(n_files):
        out_dict = {
            'points': points_uniform[file_idx],
            'occupancies': occupancies[file_idx],
        }
        np.savez(os.path.join(output_folder, sample_name, 'points_uniform_%02d.npz' % file_idx), **out_dict)
    

def create_dataset(base_path, dataset_name, output_folder, proc, num_proc):
    list_of_samples = [x.name.split('.')[0] for x in (Path(base_path) / "sdf_064" / dataset_name).iterdir()]
    list_of_samples = [x for i, x in enumerate(list_of_samples) if i % num_proc == proc]
    for p in tqdm(list_of_samples):
        try:
            create_data_point_for_sdf(Path(base_path), dataset_name, p, Path(output_folder))
        except Exception as e:
            print("DataprocessingError: ", p, e)


def create_uniform_dataset(base_path, dataset_name, output_folder, proc, num_proc):
    # list_of_samples = [x.name.split('.')[0] for x in (Path(base_path) / "sdf_064" / dataset_name).iterdir()]
    list_of_samples = list(set(Path(output_folder, "train.lst").read_text().splitlines() + Path(output_folder, "val.lst").read_text().splitlines()))
    list_of_samples = [x for i, x in enumerate(list_of_samples) if i % num_proc == proc]
    for p in tqdm(list_of_samples):
        try:
            create_uniform_samples_for_sdf(Path(base_path), dataset_name, p, Path(output_folder))
        except Exception as e:
            print("DataprocessingError: ", p, e)
            print(traceback.format_exc())


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--outputdir", type=str, default='output')
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--proc', default=0, type=int)

    args = parser.parse_args()

    create_uniform_dataset("/rhome/ysiddiqui/repatch/data", "3DFront", args.outputdir, args.proc, args.num_proc)