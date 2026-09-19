"""Enable bounded final decode in the latent-stitch delivery workflows."""
from __future__ import annotations

import json
from pathlib import Path


def migrate_ui(data):
    changed = False
    for node in data['nodes']:
        if node['type'] == 'MMH3H3StitchUpscale' and not any(i['name'] == 'streaming' for i in node['inputs']):
            node['inputs'].append(dict(name='streaming', type='COMBO', link=None, widget={'name': 'streaming'}))
            node['widgets_values'].append('auto')
            node['size'][1] += 30
            changed = True
    nodes = {n['id']: n for n in data['nodes']}
    for movie in list(nodes.values()):
        if movie['type'] != 'MMH3CreateVideo':
            continue
        image_port = next(i for i in movie['inputs'] if i['name'] == 'images')
        edge = next((e for e in data['links'] if e[0] == image_port['link']), None)
        if edge is None or nodes[edge[1]]['type'] != 'MMH3H3VideoDecode':
            continue
        decoder = nodes[edge[1]]
        outgoing = [e for e in data['links'] if e[1] == decoder['id']]
        if any(e[3] != movie['id'] for e in outgoing):
            continue  # An IMAGE consumer still needs the original full decode.
        removed = {e[0] for e in outgoing}
        audio_port = next(i for i in movie['inputs'] if i['name'] == 'audio')
        for e in data['links']:
            if e[3] == decoder['id']:
                e[3] = movie['id']
            elif e[0] == audio_port['link']:
                e[4] = 5
        movie['type'] = 'MMH3H3DecodeVideo'
        movie['properties']['Node name for S&R'] = movie['type']
        movie['inputs'] = decoder['inputs'] + [dict(name='streaming', type='COMBO', link=None,
                                                   widget={'name': 'streaming'}), audio_port]
        movie['outputs'].append(dict(name='decode_info_json', type='STRING', links=[]))
        movie['widgets_values'] = decoder['widgets_values'] + ['auto']
        movie['size'][1] = max(movie['size'][1], 220)
        data['nodes'] = [n for n in data['nodes'] if n['id'] != decoder['id']]
        data['links'] = [e for e in data['links'] if e[0] not in removed]
        changed = True
    return changed


def migrate_api(data):
    changed = False
    for node in list(data.values()):
        if node['class_type'] == 'MMH3H3StitchUpscale':
            if 'streaming' not in node['inputs']:
                node['inputs']['streaming'] = 'auto'
                changed = True
        if node['class_type'] != 'MMH3CreateVideo':
            continue
        source = node['inputs'].get('images')
        decoder = data.get(str(source[0])) if isinstance(source, list) else None
        if decoder is None or decoder['class_type'] != 'MMH3H3VideoDecode':
            continue
        if any(n is not node and any(isinstance(v, list) and v[0] == source[0] for v in n['inputs'].values())
               for n in data.values()):
            continue
        inputs = dict(decoder['inputs'], streaming='auto')
        if 'audio' in node['inputs']:
            inputs['audio'] = node['inputs']['audio']
        node.update(class_type='MMH3H3DecodeVideo', inputs=inputs)
        del data[str(source[0])]
        changed = True
    return changed


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    for name in ('mmh3_f05_latent_stitch.json', 'mmh3_f05_latent_stitch_upscale.json'):
        path = root / 'example_workflows' / name
        data = json.loads(path.read_text(encoding='utf-8'))
        if migrate_ui(data):
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path = root / 'tests/fixtures/workflows/mmh3_f05_latent_stitch_api.json'
    if path.is_file():
        data = json.loads(path.read_text(encoding='utf-8'))
        if migrate_api(data):
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
