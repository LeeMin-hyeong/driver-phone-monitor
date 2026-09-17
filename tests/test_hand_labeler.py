import hashlib
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from hand_labeler import Store, make_crop, square_box


class HandLabelerTests(unittest.TestCase):
    def test_edge_and_small_source_keep_square_inside_image(self):
        for size in ((640, 480), (80, 50)):
            for center in ((-10, -10), (0, 0), (630, 470), (900, 900)):
                box = square_box(center, 224, size)
                self.assertEqual(box[2]-box[0], box[3]-box[1])
                self.assertTrue(0 <= box[0] < box[2] <= size[0])
                self.assertTrue(0 <= box[1] < box[3] <= size[1])
                self.assertEqual(make_crop(Image.new('RGB', size), box).size, (224, 224))

    def test_save_resume_duplicate_and_split_inheritance(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'원본.png'
            image = Image.new('RGB', (640, 480), 'red')
            image.save(source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            store = Store(Path(folder)/'labels')
            store.splits[digest] = dict(split='train', subset='validation')
            record = store.add(source, digest, image, (0, 0, 224, 224), 'phone', 'left')
            other = store.add(source, digest, image, (224, 0, 448, 224), 'uncertain', 'right')
            self.assertIsNone(other['target'])
            self.assertEqual(record['subset'], other['subset'])
            self.assertEqual(record['subset'], 'validation')
            with Image.open(store.folder/record['crop']) as crop:
                self.assertEqual(crop.size, (224, 224))
                self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))
            with self.assertRaises(ValueError):
                store.add(source, digest, image, (0, 0, 224, 224), 'normal', 'left')
            resumed = Store(store.folder)
            self.assertEqual(resumed.data['records'], store.data['records'])


if __name__ == '__main__':
    unittest.main()
