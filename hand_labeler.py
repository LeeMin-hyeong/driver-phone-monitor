"""Local hand ROI labeling tool. Run with .gpu/Scripts/python.exe hand_labeler.py."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent
LABELS = ('phone', 'normal', 'uncertain')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def configure_tk():
    # Bundled Windows Tcl can fail to resolve Unicode absolute runtime paths.
    # Relative, forward-slash library paths work with the same installed runtime.
    if os.name == 'nt':
        for name, sub in (('TCL_LIBRARY', 'tcl8.6'), ('TK_LIBRARY', 'tk8.6')):
            path = Path(sys.base_prefix)/'tcl'/sub
            if path.is_dir() and name not in os.environ:
                try:
                    os.environ[name] = Path(os.path.relpath(path)).as_posix()
                except ValueError:
                    os.environ[name] = path.as_posix()


configure_tk()
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageTk


def create_window():
    return tk.Tk()


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def square_box(center, side, size):
    """Clamp a square to the EXIF-upright source image, without black borders."""
    side = max(1, min(round(side), *size))
    x = max(0, min(round(center[0]-side/2), size[0]-side))
    y = max(0, min(round(center[1]-side/2), size[1]-side))
    return (x, y, x+side, y+side)


def make_crop(image, box):
    return image.crop(box).resize((224, 224), Image.Resampling.LANCZOS)


def shared_splits():
    result = {}
    for split in ('train', 'test'):
        path = ROOT/'configs/dataset'/f'{split}.csv'
        if path.exists():
            with path.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    result[row['sha256']] = dict(split=split, subset=split)
    path = ROOT/'configs/dataset/train_validation.json'
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        for key, subset in (('fit_hashes', 'fit'), ('validation_hashes', 'validation')):
            for digest in data[key]:
                if digest in result and result[digest]['split'] == 'train':
                    result[digest]['subset'] = subset
    return result


class Store:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder/'annotations.json'
        self.data = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {
            'version': 1, 'output_size': [224, 224], 'records': [], 'reviewed': [], 'last_source': None}
        self.splits = shared_splits()

    def save(self):
        atomic_json(self.path, self.data)

    def add(self, source, digest, image, box, label, hand):
        if label not in LABELS:
            raise ValueError('Invalid label')
        for record in self.data['records']:
            if record['source_sha256'] == digest and tuple(record['box']) == tuple(box):
                raise ValueError('같은 위치가 이미 저장되어 있습니다. 목록에서 라벨을 수정하세요.')
        identifier = uuid.uuid4().hex
        relative = Path('crops')/(identifier+'.png')
        target = self.folder/relative
        target.parent.mkdir(exist_ok=True)
        make_crop(image, box).save(target)
        record = dict(id=identifier, crop=relative.as_posix(), source=str(source.resolve()),
                      source_sha256=digest, source_size=list(image.size), box=list(box),
                      coordinates='EXIF-upright pixels; right/bottom exclusive', label=label,
                      target={'phone': 1, 'normal': 0, 'uncertain': None}[label], hand=hand,
                      **self.splits.get(digest, dict(split='unassigned', subset='unassigned')))
        self.data['records'].append(record)
        self.save()
        return record


class Labeler:
    def __init__(self, root, source, output):
        self.root, self.store = root, Store(output)
        self.files, self.index, self.image, self.box = [], 0, None, None
        self.scale, self.offset = 1, (0, 0)
        self.root.title('손 ROI 라벨링 · 224 × 224')
        self.root.geometry('1280x850')
        self.root.minsize(1000, 720)
        self.label = tk.StringVar(value='phone')
        self.hand = tk.StringVar(value='unknown')
        self.side = tk.IntVar(value=224)
        self.status = tk.StringVar()
        self.caption = tk.StringVar()
        toolbar = ttk.Frame(root, padding=10)
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text='이미지 폴더 열기', command=self.choose_folder).pack(side='left')
        ttk.Button(toolbar, text='이전 ←', command=lambda: self.navigate(-1)).pack(side='left', padx=8)
        ttk.Button(toolbar, text='다음 →', command=lambda: self.navigate(1)).pack(side='left')
        ttk.Button(toolbar, text='검토 완료 후 다음', command=self.complete).pack(side='left', padx=8)
        ttk.Label(toolbar, textvariable=self.caption).pack(side='left', padx=10)
        body = ttk.Frame(root, padding=(10, 0))
        body.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(body, background='#17212b', highlightthickness=0)
        self.canvas.pack(side='left', fill='both', expand=True)
        panel = ttk.Frame(body, width=300, padding=(14, 0))
        panel.pack(side='right', fill='y')
        ttk.Label(panel, text='선택 영역 미리보기 · 224 × 224').pack(anchor='w')
        self.preview = ttk.Label(panel)
        self.preview.pack(pady=8)
        ttk.Label(panel, text='선택 범위 (원본 픽셀) / 휠로 조절').pack(anchor='w')
        self.slider = tk.Scale(panel, from_=32, to=1600, orient='horizontal',
                               variable=self.side, command=self.resize_box, length=260)
        self.slider.pack(fill='x')
        ttk.Button(panel, text='224 픽셀로 초기화', command=lambda: self.set_side(224)).pack(fill='x')
        for value, text in [('phone', '1 · phone — 이 손에 휴대폰 있음'),
                            ('normal', '2 · normal — 빈손 / 다른 물체'),
                            ('uncertain', '3 · uncertain — 판단 불가')]:
            ttk.Radiobutton(panel, text=text, variable=self.label, value=value).pack(anchor='w', pady=4)
        ttk.Label(panel, text='손 구분 (인물 기준, 선택 사항)').pack(anchor='w', pady=(8, 0))
        ttk.Combobox(panel, textvariable=self.hand, values=('unknown', 'left', 'right'),
                     state='readonly', width=24).pack(fill='x', pady=4)
        ttk.Button(panel, text='선택 영역 저장 (Enter)', command=self.save_crop).pack(fill='x', pady=8)
        ttk.Label(panel, text='현재 원본에서 저장한 샘플').pack(anchor='w')
        self.listbox = tk.Listbox(panel, height=7, exportselection=False)
        self.listbox.pack(fill='x', pady=5)
        self.listbox.bind('<<ListboxSelect>>', self.select_record)
        ttk.Button(panel, text='선택 샘플 라벨·손 구분 수정', command=self.relabel).pack(fill='x')
        ttk.Button(panel, text='선택 샘플 제외', command=self.remove_record).pack(fill='x', pady=5)
        ttk.Label(panel, text='클릭·드래그: 선택 영역 이동\n휠: 범위 변경 / 저장 크기: 항상 224²\n양손은 각각 선택해서 저장\n다음: 건너뛰기 / 검토 완료: 완료 기록', justify='left').pack(anchor='w', pady=8)
        ttk.Label(root, textvariable=self.status, padding=10, wraplength=1200).pack(fill='x')
        self.canvas.bind('<Configure>', lambda e: self.render())
        self.canvas.bind('<Button-1>', self.pick)
        self.canvas.bind('<B1-Motion>', self.pick)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.root.bind('<Return>', lambda e: self.save_crop())
        self.root.bind('<Left>', lambda e: self.navigate(-1))
        self.root.bind('<Right>', lambda e: self.navigate(1))
        for key, value in [('1', 'phone'), ('2', 'normal'), ('3', 'uncertain')]:
            self.root.bind(key, lambda e, v=value: self.label.set(v))
        self.open_folder(source)

    def choose_folder(self):
        folder = filedialog.askdirectory(title='원본 이미지 폴더')
        if folder:
            self.open_folder(Path(folder))

    def open_folder(self, folder):
        folder = Path(folder).resolve()
        self.files = sorted(p for p in folder.rglob('*') if p.suffix.lower() in EXTENSIONS
                            and not p.is_relative_to(self.store.folder))
        self.index = 0
        last = self.store.data.get('last_source')
        for i, path in enumerate(self.files):
            if str(path) == last:
                self.index = i
                break
        if self.files:
            self.load()
        else:
            self.image, self.box = None, None
            self.canvas.delete('all')
            self.listbox.delete(0, 'end')
            self.preview.configure(image='')
            self.caption.set('이미지 없음')
            self.status.set('이미지가 없습니다. 이미지 폴더 열기로 원본 폴더를 선택하세요.')

    def load(self):
        path = self.files[self.index]
        self.image, self.box = None, None
        self.preview.configure(image='')
        try:
            self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as im:
                self.image = ImageOps.exif_transpose(im).convert('RGB')
        except Exception as exc:
            self.canvas.delete('all')
            self.listbox.delete(0, 'end')
            self.status.set(f'열기 실패: {path.name} — {exc}')
            return
        self.slider.configure(to=min(self.image.size), from_=min(32, *self.image.size))
        self.side.set(min(224, *self.image.size))
        self.store.data['last_source'] = str(path)
        self.store.save()
        self.refresh_list()
        self.render()
        split = self.store.splits.get(self.digest, dict(split='unassigned', subset='unassigned'))
        self.status.set(f'{path} | {self.image.width}×{self.image.height} | {split["split"]}/{split["subset"]} | 저장 위치: {self.store.folder}')

    def refresh_list(self):
        self.records = [r for r in self.store.data['records'] if r['source_sha256'] == self.digest]
        self.listbox.delete(0, 'end')
        for r in self.records:
            self.listbox.insert('end', f'{r["label"]} / {r["hand"]} / {r["box"][:2]}')
        done = ' · 검토 완료' if self.digest in self.store.data['reviewed'] else ''
        self.caption.set(f'{self.index+1} / {len(self.files)} · 저장 {len(self.records)}개{done}')

    def render(self):
        if self.image is None:
            return
        w, h = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.scale = min(w/self.image.width, h/self.image.height)
        size = (max(1, round(self.image.width*self.scale)), max(1, round(self.image.height*self.scale)))
        self.offset = ((w-size[0])/2, (h-size[1])/2)
        self.photo = ImageTk.PhotoImage(self.image.resize(size, Image.Resampling.BILINEAR))
        self.canvas.delete('all')
        self.canvas.create_image(*self.offset, image=self.photo, anchor='nw')
        for r in self.records:
            self.draw_box(r['box'], '#66baff', r['label'])
        if self.box:
            self.draw_box(self.box, '#ffcf40', '선택 / selection')
            self.crop_photo = ImageTk.PhotoImage(make_crop(self.image, self.box))
            self.preview.configure(image=self.crop_photo)

    def draw_box(self, box, color, text):
        x1, y1, x2, y2 = [v*self.scale+self.offset[i % 2] for i, v in enumerate(box)]
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
        self.canvas.create_text(x1+3, y1+3, text=text, fill=color, anchor='nw')

    def pick(self, event):
        if self.image is None:
            return
        center = ((event.x-self.offset[0])/self.scale, (event.y-self.offset[1])/self.scale)
        self.box = square_box(center, self.side.get(), self.image.size)
        self.render()

    def resize_box(self, _=None):
        if self.image is not None and self.box:
            center = ((self.box[0]+self.box[2])/2, (self.box[1]+self.box[3])/2)
            self.box = square_box(center, self.side.get(), self.image.size)
            self.render()

    def set_side(self, side):
        if self.image is not None:
            self.side.set(max(min(32, *self.image.size), min(side, *self.image.size)))
            self.resize_box()

    def wheel(self, event):
        self.set_side(self.side.get() + (16 if event.delta > 0 else -16))

    def save_crop(self):
        if self.image is None or not self.box:
            self.status.set('먼저 원본에서 손 주변을 클릭해 선택하세요.')
            return
        try:
            self.store.add(self.files[self.index], self.digest, self.image, self.box, self.label.get(), self.hand.get())
        except Exception as exc:
            messagebox.showerror('저장 실패', str(exc))
            return
        self.box = None
        self.preview.configure(image='')
        self.refresh_list()
        self.render()
        self.status.set('저장 완료. 반대쪽 손도 선택해 저장하거나 검토 완료 후 다음으로 이동하세요.')

    def selected(self):
        selection = self.listbox.curselection()
        return self.records[selection[0]] if selection else None

    def select_record(self, _=None):
        record = self.selected()
        if record:
            self.box = tuple(record['box'])
            self.side.set(self.box[2]-self.box[0])
            self.label.set(record['label'])
            self.hand.set(record['hand'])
            self.render()

    def relabel(self):
        record = self.selected()
        if record:
            record.update(label=self.label.get(), target={'phone': 1, 'normal': 0, 'uncertain': None}[self.label.get()], hand=self.hand.get())
            self.store.save()
            self.refresh_list()
            self.render()

    def remove_record(self):
        record = self.selected()
        if record:
            # Keep the crop on disk for recovery; only active manifest records are training samples.
            self.store.data['records'].remove(record)
            self.store.save()
            self.box = None
            self.preview.configure(image='')
            self.refresh_list()
            self.render()
            self.status.set('목록에서 제외했습니다. 복구를 위해 crop 파일은 디스크에 유지됩니다.')

    def navigate(self, step):
        if self.files:
            self.index = max(0, min(self.index+step, len(self.files)-1))
            self.load()

    def complete(self):
        if self.image is not None:
            if self.digest not in self.store.data['reviewed']:
                self.store.data['reviewed'].append(self.digest)
            self.store.save()
            self.navigate(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'dataset/train')
    parser.add_argument('--output', type=Path, default=ROOT/'dataset/hand_rois')
    args = parser.parse_args()
    root = create_window()
    Labeler(root, args.source, args.output)
    root.mainloop()


if __name__ == '__main__':
    main()
