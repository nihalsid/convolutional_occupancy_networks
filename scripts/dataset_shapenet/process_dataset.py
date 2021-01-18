from pathlib import Path
import struct
import marching_cubes as mc
import numpy as np
import torch
from tqdm import tqdm


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


def get_sdf_paths(sample_name):
    return sample_name + "_frame0.sdf", sample_name + "_frame0.colors", sample_name + ".sdf", sample_name + ".colors"

def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], :] = values
    if nf_values > 1:
        return dense.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])


def parse_sdf_and_color_files(sdf_path, color_path):
    fin = open(sdf_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
    # sdf
    num = struct.unpack('Q', fin.read(8))[0]
    locs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
    locs = np.asarray(locs, dtype=np.int32).reshape([num, 3])
    locs = np.flip(locs, 1).copy()  # convert to zyx ordering
    sdfs = struct.unpack('f' * num, fin.read(num * 4))
    sdfs = np.asarray(sdfs, dtype=np.float32).reshape((-1, 1))
    sdfs /= voxelsize
    sdfs = sparse_to_dense_np(locs, sdfs, dimx, dimy, dimz, '-inf')

    fin = open(color_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    # color
    num = struct.unpack('Q', fin.read(8))[0]
    colors = struct.unpack('B' * num * 3, fin.read(num * 3))
    colors = np.array(colors, dtype=np.uint8).reshape(num, 3)
    colors = sparse_to_dense_np(locs, colors, dimx, dimy, dimz, 0)
    colors = colors.astype(np.float32)
    return sdfs, colors, np.array(world2grid).astype(np.float32)


def load_sdf_colors_file(base_path, sample_name):
    trunc = 3.
    in_sdf_name, in_col_name, tgt_sdf_name, tgt_col_name = get_sdf_paths(sample_name)
    in_sdf, in_colors, world2grid = parse_sdf_and_color_files(base_path / in_sdf_name, base_path / in_col_name)
    tgt_sdf, tgt_colors, world2grid = parse_sdf_and_color_files(base_path / tgt_sdf_name, base_path / tgt_col_name)    
    in_sdf[np.greater(np.abs(in_sdf), trunc)] = trunc
    tgt_sdf[np.greater(np.abs(tgt_sdf), trunc)] = trunc
    in_colors[np.greater(np.abs(in_sdf), trunc)] = 0
    tgt_colors[np.greater(np.abs(tgt_sdf), trunc)] = 0
    tgt_occ = np.less(np.abs(tgt_sdf), trunc)
    in_sdf = torch.nn.functional.interpolate(torch.from_numpy(in_sdf).float().cuda().unsqueeze(0).unsqueeze(0), recompute_scale_factor=True, scale_factor=0.5).squeeze().cpu().numpy()
    in_colors = torch.nn.functional.interpolate(torch.from_numpy(in_colors).float().cuda().unsqueeze(0).permute((0, 4, 1, 2, 3)), recompute_scale_factor=True, scale_factor=0.5).permute((0, 2, 3, 4, 1)).squeeze().cpu().numpy()
    return in_sdf, in_colors, tgt_sdf, tgt_colors, tgt_occ


def create_data_point_for_sdf(base_path, sample_name, output_folder):
    in_sdf, in_colors, tgt_sdf, tgt_colors, tgt_occ = load_sdf_colors_file(base_path, sample_name)
    # sample points for target occupancies
    # also include its colors
    # export iou points 
    # export input color and sdf volumes
    points_uniform_ratio = 0.05
    dim = 96
    points_sigma = 0.01

    occupied_pts_surface_true = (np.stack(np.where(tgt_sdf <= 0.50)).T / dim) - 0.5
    occupied_pts_surface = (np.stack(np.where(tgt_sdf <= 0.75)).T / dim) - 0.5
    iou_points = (np.stack(np.where(tgt_sdf <= 1.5)).T / dim) - 0.5
    color_eval_points = (np.stack(np.where(tgt_sdf <= 0.75)).T / dim) - 0.5
    # visualize_point_list(np.stack(np.where(tgt_sdf <= 0.75)).T, None, "occ.obj")

    n_points_uniform = int(occupied_pts_surface.shape[0] * (points_uniform_ratio))

    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = (points_uniform - 0.5)
    points_surface = occupied_pts_surface
    points_surface += points_sigma * np.random.randn(points_surface.shape[0], 3)
    points_surface_true = occupied_pts_surface_true
    points = np.concatenate([points_uniform, points_surface, points_surface_true], axis=0)

    points_grid = np.clip(((points + 0.5) * dim), 0, dim - 1).astype(np.uint32)
    occupancies = tgt_sdf[points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]] <= 0.75
    colors = tgt_colors[points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]]

    iou_points_grid = np.clip(((iou_points + 0.5) * dim), 0, dim - 1).astype(np.uint32)
    iou_occupancies = tgt_sdf[iou_points_grid[:, 0], iou_points_grid[:, 1], iou_points_grid[:, 2]] <= 0.75

    color_eval_points_grid = np.clip(((color_eval_points + 0.5) * dim), 0, dim - 1).astype(np.uint32)
    color_eval_colors = tgt_colors[color_eval_points_grid[:, 0], color_eval_points_grid[:, 1], color_eval_points_grid[:, 2]]

    # visualize_point_list(iou_points[iou_occupancies==True, :], None, "points.obj")
    # visualize_point_list(color_eval_points, color_eval_colors, "colors.obj")
    # visualize_point_list(points_grid[occupancies==True, :], colors[occupancies==True, :], "occtest.obj")
    # visualize_sdf_color(in_sdf, in_colors, "incomplete.obj")
    
    dtype = np.float16
    points = points.astype(dtype)
    colors = colors.astype(dtype)
    iou_points = iou_points.astype(dtype)

    (output_folder / sample_name).mkdir(exist_ok=True, parents=True)
    
    np.savez_compressed((output_folder / sample_name / "points.npz"), points=points, occupancies=occupancies, colors=colors)
    np.savez_compressed((output_folder / sample_name / "points_iou.npz"), points=iou_points, occupancies=iou_occupancies)
    np.savez_compressed((output_folder / sample_name / "color_eval.npz"), points=color_eval_points, colors=color_eval_colors)
    np.savez_compressed((output_folder / sample_name / "tsdf.npz"), geometry=in_sdf, color=in_colors)    


def create_dataset(base_path, output_folder):
    list_of_samples = []
    for p in tqdm([x.name.split('.')[0] for x in Path(base_path).iterdir() if x.name.endswith("__0__.sdf")]):
        create_data_point_for_sdf(Path(base_path), p, Path(output_folder))
        list_of_samples.append(p)
    (Path(output_folder) / "all.lst").write_text("\n".join(list_of_samples))


if __name__ == "__main__":
    create_dataset("shapenet_tsdf_sample", "../../data/demo/shapenet_chair/null_category")