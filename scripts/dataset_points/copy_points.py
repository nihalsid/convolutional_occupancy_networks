from pathlib import Path
from shutil import copyfile
from tqdm import tqdm


def copy_points(src_dir, dest_dir):
    skipped = 0
    copied = 0
    src_dir_iterated = list(src_dir.iterdir())
    for pc in tqdm(src_dir_iterated):
        dest = dest_dir / pc.name.split('.')[0] / "pointcloud.npz"
        if (dest_dir / pc.name.split('.')[0]).exists():
            copyfile(pc, dest)
            copied += 1
        else:
            skipped += 1
    print(f"Skipped: {skipped} / {len(src_dir_iterated)}, copied = {copied}")


def copy_tsdf(src_dir, dest_dir):
    skipped = 0
    copied = 0
    src_dir_iterated = [x for x in list(src_dir.iterdir()) if x.name.endswith(".npz")]
    for pc in tqdm(src_dir_iterated):
        dest = dest_dir / pc.name.split('.')[0] / "target_tsdf.npz"
        if (dest_dir / pc.name.split('.')[0]).exists():
            copyfile(pc, dest)
            copied += 1
        else:
            skipped += 1
    print(f"Skipped: {skipped} / {len(src_dir_iterated)}, copied = {copied}")


if __name__ == "__main__":
    import sys
    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    copy_tsdf(Path(src_dir), Path(dest_dir))
