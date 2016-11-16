from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote
from urllib.error import HTTPError, URLError


def get_tok(data):
    url = 'https://openapi.baidu.com/oauth/2.0/token'
    data['grant_type'] = 'client_credentials'
    data_encoded = urlencode(data).encode('utf-8')
    try:
        req = Request(url, data=data_encoded)
        rsp = urlopen(req, timeout=5).read().decode('utf-8')
    except (HTTPError, URLError) as e:
        # print(e)
        return None
    except Exception as e:
        # print(e)
        return None
    import json
    return json.loads(rsp)['access_token']


def get_audio(data):
    url = 'http://tsn.baidu.com/text2audio'
    data_encoded = urlencode(data).encode('utf-8')
    try:
        req = Request(url, data=data_encoded)
        rsp = urlopen(req, timeout=5)
    except (HTTPError, URLError) as e:
        # print(e)
        return None
    except Exception as e:
        # print(e)
        return None
    if rsp.getheader('Content-Type') == 'application/json':
        # print(rsp.read().encode('utf-8'))
        return None
    return rsp.read()


def tts(tok_params, tex_str, aud_params=None):
    aud_params = aud_params or {
        'lan': 'zh',
        'cuid': '1234567',
        'ctp': 1,
        'spd': 7,
        'pit': 9,
        'vol': 9,
    }
    aud_params['tex'] = tex_str
    tok = get_tok(tok_params)
    aud_params['tok'] = tok
    rsp = get_audio(aud_params)
    return rsp


def save_sound(sound, output_filename):
    if sound:
        with open(output_filename, 'wb') as f:
            f.write(sound)
    else:
        print(sound)


def play_sound(audio_filename, values=''):
    import os
    os.system('play {0} {1}'.format(audio_filename, values))
