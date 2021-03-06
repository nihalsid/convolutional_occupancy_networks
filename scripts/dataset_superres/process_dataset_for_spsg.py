from pathlib import Path
import struct
import marching_cubes as mc
import numpy as np
import torch
from tqdm import tqdm
import trimesh


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


def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], :] = values
    if nf_values > 1:
        return dense.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])


def load_sdf(file_path):
    fin = open(file_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f'*4*4, fin.read(4*4*4))
    world2grid = np.asarray(world2grid, dtype=np.float32).reshape([4, 4])
    num = struct.unpack('Q', fin.read(8))[0]
    locs = struct.unpack('I'*num*3, fin.read(num*3*4))
    locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
    locs = np.flip(locs, 1).copy() # convert to zyx ordering
    sdf = struct.unpack('f'*num, fin.read(num*4))
    sdf = np.asarray(sdf, dtype=np.float32)
    sdf /= voxelsize
    sdf = sparse_to_dense_np(locs, sdf[:,np.newaxis], dimx, dimy, dimz, -float('inf'))
    sdf[np.abs(sdf) > 3] = 3
    sdf = sdf.transpose((2, 1, 0))
    fin.close()
    return sdf


def create_data_point_for_sdf(base_path, sample_name, output_folder):
    in_sdf = load_sdf(Path(base_path, sample_name + ".sdf"))[:96, :96, :96]
    tgt_sdf = load_sdf(Path(base_path, sample_name.replace("__inc__", "__cmp__") + ".sdf"))[:96, :96, :96]
    
    points_uniform_ratio = 0.5
    dim = np.array(in_sdf.shape)
    points_sigma_0 = 0.01
    points_sigma_2 = 0.1
    points_sigma_3 = 0.05

    occupied_pts_surface_true = (np.stack(np.where(np.abs(tgt_sdf) <= 0.75)).T / dim) - 0.5
    points_vicinity = (np.stack(np.where((np.abs(tgt_sdf) <= 1) & (np.abs(tgt_sdf) > 0.75))).T / dim) - 0.5
    iou_points = (np.stack(np.where(np.abs(tgt_sdf) <= 1.5)).T / dim) - 0.5
    
    points_surface_sigma_0 = occupied_pts_surface_true + points_sigma_0 * np.random.randn(occupied_pts_surface_true.shape[0], 3)
    points_surface_sigma_2 = occupied_pts_surface_true + points_sigma_2 * np.random.randn(occupied_pts_surface_true.shape[0], 3)
    points_surface_sigma_3 = occupied_pts_surface_true + points_sigma_3 * np.random.randn(occupied_pts_surface_true.shape[0], 3)

    points_surface_true = occupied_pts_surface_true
    n_points_uniform = int((points_surface_true.shape[0] * 3)  * (points_uniform_ratio))
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = (points_uniform - 0.5)
    points = np.concatenate([points_surface_sigma_0, points_surface_sigma_2, points_surface_sigma_3, points_uniform, points_surface_true], axis=0)
    points_mask = ((points > -0.5) & (points < 0.5)).all(axis=1)
    points = points[points_mask, :]

    points_grid = ((points + 0.5) * dim).astype(np.uint32)
    points_grid_inside = ((points_grid >=0) & (points_grid < dim)).all(axis=1)
    occupancies = np.zeros((points_grid.shape[0]), dtype=np.bool)
    occupancies[points_grid_inside] = tgt_sdf[points_grid[points_grid_inside, 0], points_grid[points_grid_inside, 1], points_grid[points_grid_inside, 2]] <= 0.75

    iou_points_grid = np.clip(((iou_points + 0.5) * dim), 0, dim - 1).astype(np.uint32)
    iou_occupancies = np.abs(tgt_sdf[iou_points_grid[:, 0], iou_points_grid[:, 1], iou_points_grid[:, 2]]) <= 0.75

    # visualize_point_list(iou_points[iou_occupancies==True, :], None, f"dump/{sample_name}_iou_pts_true.obj")
    # visualize_point_list(iou_points, None, f"dump/{sample_name}_iou_pts.obj")
    # visualize_point_list(points[occupancies==True, :], None, f"dump/{sample_name}_occ.obj")
    # visualize_point_list(points, None, f"dump/{sample_name}_samples.obj")
    # visualize_probs(np.abs(in_sdf) <= 0.75, f"dump/{sample_name}_insdf.obj")

    dtype = np.float16
    points = points.astype(dtype)
    iou_points = iou_points.astype(dtype)
    if points.shape[0] > 0:
        (output_folder / sample_name.replace('__inc__', '_')).mkdir(exist_ok=True, parents=True)
        np.savez_compressed((output_folder / sample_name.replace('__inc__', '_') / "points.npz"), points=points, occupancies=occupancies)
        np.savez_compressed((output_folder / sample_name.replace('__inc__', '_') / "points_iou.npz"), points=iou_points, occupancies=iou_occupancies)
        np.savez_compressed((output_folder / sample_name.replace('__inc__', '_') / "tsdf.npz"), geometry=torch.nn.functional.interpolate(torch.from_numpy(in_sdf).unsqueeze(0).unsqueeze(0), scale_factor=0.5, recompute_scale_factor=True).squeeze(0).squeeze(0).numpy())    


def create_dataset(base_path, output_folder, proc, num_proc):
    list_of_samples = [x.name.split('.')[0] for x in (Path(base_path)).iterdir() if '__inc__' in x.name]
    list_of_samples = [x for i, x in enumerate(list_of_samples) if i % num_proc == proc]
    for p in tqdm(list_of_samples):
        create_data_point_for_sdf(Path(base_path), p, Path(output_folder))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sdfdir", type=str, default='/mnt/pegasus_raid_yawar/individual_96-96-160')
    parser.add_argument("--outputdir", type=str, default='output')
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--proc', default=0, type=int)

    args = parser.parse_args()

    create_dataset(args.sdfdir, args.outputdir, args.proc, args.num_proc)