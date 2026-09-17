"""Verify the shared train/test file list, labels, contents, and separation."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def main():
    hashes = {}
    for split in ('train', 'test'):
        folder = (ROOT/'dataset'/split).resolve()
        with (ROOT/'configs/dataset'/f'{split}.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        expected = set()
        split_hashes = set()
        for row in rows:
            path = (ROOT/row['path']).resolve()
            if not path.is_relative_to(folder):
                raise ValueError(f'Path outside {split}: {row["path"]}')
            if row['label'] not in ('phone', 'normal') or int(row['target']) != int(row['label'] == 'phone'):
                raise ValueError(f'Invalid label: {row["path"]}')
            if path.relative_to(folder).parts[0] != row['label']:
                raise ValueError(f'Wrong label directory: {row["path"]}')
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Image content changed: {row["path"]}')
            expected.add(path)
            split_hashes.add(row['sha256'])
        actual = {p.resolve() for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS}
        if expected != actual or len(expected) != len(rows) or len(split_hashes) != len(rows):
            raise ValueError(f'{split}: missing, added, or duplicate images')
        hashes[split] = split_hashes
        print(f'{split}: {len(rows)} images match the shared manifest')
    if hashes['train'] & hashes['test']:
        raise ValueError('Train/test exact-file overlap')
    print('PASS: same shared dataset, labels and split; no exact-file overlap')


if __name__ == '__main__':
    main()
