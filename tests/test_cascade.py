import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from cascade import driver_crop, IMAGE_SIZE


class DriverCropTests(unittest.TestCase):
    def test_no_person_uses_configured_seat_side(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cabin.png'
            image = Image.new('RGB', (200,100), 'blue')
            ImageDraw.Draw(image).rectangle((100,0,199,99), fill='red')
            image.save(path)
            for side, expected in [('right',(255,0,0)),('left',(0,0,255))]:
                crop = driver_crop(path, dict(boxes=[]), side)
                self.assertEqual(crop.size, (IMAGE_SIZE, IMAGE_SIZE))
                self.assertEqual(crop.getpixel((IMAGE_SIZE//2,IMAGE_SIZE//2)), expected)

    def test_person_box_is_used_without_pose(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'solo.png'
            image = Image.new('RGB', (200,100), 'blue')
            ImageDraw.Draw(image).rectangle((0,0,70,99), fill='red')
            image.save(path)
            obs = dict(boxes=[dict(cls=0, confidence=.9, box=[0,0,.3,1])], poses=[])
            crop = driver_crop(path, obs, 'right')
            self.assertEqual(crop.getpixel((IMAGE_SIZE//2,IMAGE_SIZE//2)), (255,0,0))


if __name__ == '__main__':
    unittest.main()
