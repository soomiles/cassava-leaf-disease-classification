import shutil
from pathlib import Path
import argparse

def main(args):
    get_size = lambda path: sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())

    root_paths = Path(args.root_dir).glob('*')

    for root_path in root_paths:
        for path in root_path.glob('*'):
            if not path.is_dir(): continue
            kb_size = get_size(path) / (1024 ** 2)

            if kb_size < 200.:
                shutil.rmtree(path)
                print(f"{path.parent.name}/{path.name}: {kb_size:.4f} kb (deleted)")
            else:
                print(f"{path.parent.name}/{path.name}: {kb_size:.4f} kb")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='root path')

    parser.add_argument('--root-dir', type=str,
                        default='/workspace/logs/cassava-leaf-disease-classification/',
                        help='root log directory')

    args = parser.parse_args()
    main(args)