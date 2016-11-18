"""
Configuration of CartoonPy


This file enables you to specify a configuration for CartoonPy. In
particular you can enter the path to the FFMPEG and ImageMagick
binaries.

Defaults must be done BEFORE installing CartoonPy: first make the changes,
then install CartoonPy with

    [sudo] python setup.py install

Note that you can also change the path by setting environment variables.
e.g.

Linux/Mac:
   export FFMPEG_BINARY=path/to/ffmpeg

Windows:
   set FFMPEG_BINARY=path\to\ffmpeg

Instructions
--------------

FFMPEG_BINARY
    Normally you can leave this one to its default ('ffmpeg-imageio') at which
    case image-io will download the right ffmpeg binary (at first use) and then
    always use that binary.
    The second option is 'auto-detect', in this case ffmpeg will be whatever
    binary is found on the computer generally 'ffmpeg' (on linux) or 'ffmpeg.exe'
    (on windows).
    Third option: If you want to use a binary at a special location on you disk,
    enter it like that:

    FFMPEG_BINARY = r"path/to/ffmpeg" # on linux
    FFMPEG_BINARY = r"path\to\ffmpeg.exe" # on windows

    Warning: the 'r' before the path is important, especially on Windows.


TTS_TOK_PARAMS
    For linux users, 'CLIENT_ID' and 'CLIENT_SECRET' can be write in ~/.bashrc
    For Windows users, you can set it by Windows manage tools

"""

import os

FFMPEG_BINARY = os.getenv('FFMPEG_BINARY', 'ffmpeg-imageio')

TTS_TOK_PARAMS = {
    'client_id': os.getenv('TTS_CLIENT_ID'),
    'client_secret': os.getenv('TTS_CLIENT_SECRET'),
}

AUDIO_PARAMS = {
    'lan': 'zh',
    'cuid': 'bxtkezhan',
    'ctp': 1,
    'spd': 7,
    'pit': 5,
    'vol': 9,
    'per': 0,
}
