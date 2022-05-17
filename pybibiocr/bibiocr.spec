# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(['bibiocr.py'],
     pathex=[],
     binaries=[],
     datas=[
            ('onnx/models/','onnx/models'),
            ('*.json','.'),
            ('*.png','.'),
            ('*.ico','.'),
            ('build/lib.win-amd64-3.7/pybibiocr/mainwindow.pyd','.')],
     hiddenimports=[ 'pyqt5',
                     'pyqt5-tools',
                     'shapely',
                     'pyclipper',
                     'onnxruntime',
                     'pillow',
                     'imageio',
                     'imghdr',
                     'mainwindow'
            ],
     hookspath=['.'],
     runtime_hooks=[],
     excludes=['matplotlib'],
     win_no_prefer_redirects=False,
     win_private_assemblies=False,
     cipher=block_cipher,
     noarchive=False)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz,
     a.scripts,
     [],
     exclude_binaries=True,
     name='bibiocr',
     debug=False,
     bootloader_ignore_signals=False,
     strip=False,
     upx=True,
#     console=True,
     console=False,
     icon='bibiocr.ico')

coll = COLLECT(exe,
     a.binaries,
     a.zipfiles,
     a.datas,
     strip=False,
     upx=True,
     upx_exclude=[],
     name='bibiocr')