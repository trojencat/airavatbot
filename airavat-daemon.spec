# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Airavat Daemon
# Build with: pyinstaller airavat-daemon.spec

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os
import sys

block_cipher = None

# Collect data files and submodules for dependencies
mcp_datas = collect_data_files('mcp')
fastapi_datas = collect_data_files('fastapi')
pydantic_datas = collect_data_files('pydantic')

# Webui directory (relative to spec file)
webui_dir = os.path.join(os.path.dirname(SPEC), 'webui')

a = Analysis(
    ['__main__.py'],
    pathex=[],
    binaries=[],
    datas=[
        (webui_dir, 'webui') if os.path.isdir(webui_dir) else None,
    ] + mcp_datas + fastapi_datas + pydantic_datas,
    hiddenimports=[
        'src',
        'src.server',
        'src.config',
        'src.agent',
        'src.mcp_manager',
        'mcp',
        'mcp.server',
        'mcp.types',
        'anthropic',
        'ollama',
        'fastapi',
        'uvicorn',
        'pydantic',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludedimports=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove None entries from datas
a.datas = [d for d in a.datas if d is not None]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='airavat-daemon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
