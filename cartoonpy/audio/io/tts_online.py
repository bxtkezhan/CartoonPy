from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote
from urllib.error import HTTPError, URLError
import json


def get_rst(url, data):
    data_encoded = urlencode(data).encode('utf-8')
    try:
        req = Request(url, data=data_encoded)
        rst = urlopen(req, timeout=5)
    except (HTTPError, URLError) as e:
        # print(e)
        return None
    except Exception as e:
        # print(e)
        return None
    return rst


def get_tok(data):
    url = 'https://openapi.baidu.com/oauth/2.0/token'
    data['grant_type'] = 'client_credentials'
    rst = get_rst(url, data)
    if rst is None: return None
    return json.loads(rst.read().decode())['access_token']


def get_audio(data):
    url = 'http://tsn.baidu.com/text2audio'
    data_encoded = urlencode(data).encode('utf-8')
    rst = get_rst(url, data)
    if rst is None or rst.getheader('Content-Type') == 'application/json':
        # print(rsp.read().encode('utf-8'))
        return None
    return rst.read()


def tts(tok_params, tex_str, aud_params=None):
    aud_params = aud_params or {
        'lan': 'zh',
        'cuid': 'bxtkezhan',
        'ctp': 1,
        'spd': 7,
        'pit': 5,
        'vol': 9,
        'per': 0,
    }
    aud_params['tex'] = tex_str
    tok = get_tok(tok_params)
    if tok:
        aud_params['tok'] = tok
        aud = get_audio(aud_params)
        return aud


def save_sound(sound, output_filename):
    if sound:
        with open(output_filename, 'wb') as f:
            f.write(sound)
    else:
        print(sound)
