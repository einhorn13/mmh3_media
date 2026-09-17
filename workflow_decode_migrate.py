"""Migrate final decode controls and wire explicit video decode provenance."""
import copy
import json
import uuid
from pathlib import Path

MODES = ['vae', 'draft', 'trt']


def migrate_api(data):
    for node in data.values():
        if not isinstance(node, dict):
            continue
        if node.get('class_type') == 'BasicScheduler':
            node['class_type'] = 'MMH3H3Scheduler'
        if node.get('class_type') == 'MMH3H3VideoDecode':
            values = node['inputs']
            old = values.pop('draft_tae', False)
            values.setdefault('decode_mode', 'draft' if old is True else 'vae')
            values.setdefault('trt_decoder', 'auto')
    for node in data.values():
        if not isinstance(node, dict) or node.get('class_type') != 'CreateVideo':
            continue
        source = node['inputs'].get('images')
        if isinstance(source, list) and data.get(source[0], {}).get('class_type') == 'MMH3H3VideoDecode':
            node['class_type'] = 'MMH3CreateVideo'
            node['inputs']['decode_info_json'] = [source[0], 1]


def migrate_recipe(data):
    scopes = {'root': data['graph'], **data.get('subgraphs', {})}
    for name, scope in scopes.items():
        nodes = scope['nodes']
        for key, node in list(nodes.items()):
            if node['type'] == 'BasicScheduler':
                node['type'] = 'MMH3H3Scheduler'
            if node['type'] == 'MMH3H3VideoDecode':
                values, bindings = node['values'], node['bindings']
                old = values.pop('draft_tae', False)
                values.setdefault('decode_mode', 'draft' if old else 'vae')
                values.setdefault('trt_decoder', 'auto')
                if 'draft_tae' in bindings:
                    bindings.pop('draft_tae')
                    bindings['decode_mode'] = ['$in', 'decode_mode']
                    bindings['trt_decoder'] = ['$in', 'trt_decoder']
        for port in scope.get('inputs', []):
            if port['name'] == 'draft_tae':
                port.update(name='decode_mode', type='COMBO', options=MODES, default='vae', force_input=False)
                scope['inputs'].append(dict(id=str(uuid.uuid4()), name='trt_decoder', type='COMBO',
                                            options=['auto'], default='auto', force_input=False))
                break
        for key, node in list(nodes.items()):
            values, bindings = node.get('values', {}), node.get('bindings', {})
            if 'draft_tae' in bindings and node['type'] in scopes:
                old_source = bindings.pop('draft_tae')[0]
                values.update(decode_mode='vae', trt_decoder='auto')
                if old_source in nodes and nodes[old_source]['type'] == 'PrimitiveBoolean':
                    del nodes[old_source]
            if node['type'] == 'MMH3H3StitchUpscale':
                values.setdefault('decode_mode', 'vae')
                values.setdefault('trt_decoder', 'auto')
            if node['type'] == 'CreateVideo':
                source = bindings.get('images')
                if source and source[0] in nodes and nodes[source[0]]['type'] == 'MMH3H3VideoDecode':
                    node['type'] = 'MMH3CreateVideo'
                    bindings['decode_info_json'] = [source[0], 'decode_info_json']
        for node in nodes.values():
            props = node.get('layout', {}).get('properties', {})
            if 'Node name for S&R' in props:
                props['Node name for S&R'] = node['type']


def migrate_ui(data):
    scopes = [data] + data.get('definitions', {}).get('subgraphs', [])
    for scope in scopes:
        inside = scope is not data
        nodes = {n['id']: n for n in scope['nodes']}

        def add_link(origin, oslot, target, tslot, kind):
            lid = max([(l['id'] if isinstance(l, dict) else l[0]) for l in scope['links']] + [0]) + 1
            link = dict(id=lid, origin_id=origin, origin_slot=oslot, target_id=target, target_slot=tslot, type=kind)
            scope['links'].append(link if inside else [lid, origin, oslot, target, tslot, kind])
            if origin in nodes:
                nodes[origin]['outputs'][oslot].setdefault('links', []).append(lid)
            if target in nodes:
                nodes[target]['inputs'][tslot]['link'] = lid
            if inside:
                scope['state']['lastLinkId'] = lid
            else:
                scope['last_link_id'] = lid
            return lid

        for node in nodes.values():
            if node['type'] == 'BasicScheduler':
                node['type'] = 'MMH3H3Scheduler'
            if node['type'] == 'MMH3H3Scheduler' and len(node.get('widgets_values', [])) == 1:
                node['widgets_values'] = ['simple', 20, node['widgets_values'][0]]
            if node['type'] == 'MMH3H3VideoDecode' and any(p['name'] == 'draft_tae' for p in node['inputs']):
                port = next(p for p in node['inputs'] if p['name'] == 'draft_tae')
                port.update(name='decode_mode', type='COMBO', widget={'name': 'decode_mode'})
                old = node.get('widgets_values', [False])[0]
                node['widgets_values'] = ['draft' if old else 'vae', 'auto']
                node['inputs'].append(dict(name='trt_decoder', type='COMBO', link=None, widget={'name': 'trt_decoder'}))
                node['outputs'].append(dict(name='decode_info_json', type='STRING', links=[]))
                node['size'][1] = max(node['size'][1], 190)
                if inside and port.get('link') is not None:
                    link = next(l for l in scope['links'] if l['id'] == port['link'])
                    link['type'] = 'COMBO'
                    index = link['origin_slot']
                    public = scope['inputs'][index]
                    public.update(name='decode_mode', type='COMBO')
                    trt_index = len(scope['inputs'])
                    lid = add_link(-10, trt_index, node['id'], 3, 'COMBO')
                    scope['inputs'].append(dict(id=str(uuid.uuid5(uuid.UUID(scope['id']), 'trt_decoder')),
                                                name='trt_decoder', type='COMBO', linkIds=[lid], pos=[-180, 40+trt_index*20]))
                    for instance in data['nodes']:
                        if instance['type'] != scope['id']:
                            continue
                        old_port = instance['inputs'][index]
                        old_link = old_port.get('link')
                        if old_link is not None:
                            root_link = next(l for l in data['links'] if l[0] == old_link)
                            primitive = next(n for n in data['nodes'] if n['id'] == root_link[1])
                            if primitive['type'] != 'PrimitiveBoolean':
                                raise ValueError('Non-primitive draft control requires explicit migration')
                            data['links'].remove(root_link)
                            primitive['outputs'][root_link[2]]['links'].remove(old_link)
                            if not any(o.get('links') for o in primitive['outputs']):
                                primitive['_remove_decode_toggle'] = True
                        old_port.update(name='decode_mode', type='COMBO', link=None, widget={'name': 'decode_mode'})
                        instance['inputs'].append(dict(name='trt_decoder', type='COMBO', link=None, widget={'name': 'trt_decoder'}))
                        # Subgraph widgets are synthesized from their interior widget defaults.
                        instance['widgets_values'] = []
                        instance['size'][1] += 24
            if node['type'] == 'MMH3H3StitchUpscale' and not any(p['name'] == 'decode_mode' for p in node['inputs']):
                # New optional widgets precede force_unload in the runtime schema.
                index = next(i for i,p in enumerate(node['inputs']) if p['name'] == 'force_unload')
                node['inputs'][index:index] = [dict(name=n, type='COMBO', link=None, widget={'name':n}) for n in ('decode_mode','trt_decoder')]
                node['widgets_values'][-1:-1] = ['vae','auto']
                for l in scope['links']:
                    if inside and l['target_id'] == node['id'] and l['target_slot'] >= index:
                        l['target_slot'] += 2
                    elif not inside and l[3] == node['id'] and l[4] >= index:
                        l[4] += 2
        # A decoder may export IMAGE from a subgraph before CreateVideo at root.
        for decoder in list(nodes.values()):
            if decoder['type'] != 'MMH3H3VideoDecode':
                continue
            for link in list(scope['links']):
                origin, oslot, target, tslot = ((link[k] for k in ('origin_id','origin_slot','target_id','target_slot')) if inside else link[1:5])
                if (origin, oslot) != (decoder['id'], 0):
                    continue
                if inside and target == -20:
                    for instance in data['nodes']:
                        if instance['type'] != scope['id']:
                            continue
                        for root_link in list(data['links']):
                            if root_link[1:3] != [instance['id'], tslot]:
                                continue
                            movie = next(n for n in data['nodes'] if n['id'] == root_link[3])
                            if movie['type'] != 'CreateVideo' or movie['inputs'][root_link[4]]['name'] != 'images':
                                continue
                            slot = len(scope['outputs'])
                            lid = add_link(decoder['id'], 1, -20, slot, 'STRING')
                            scope['outputs'].append(dict(id=str(uuid.uuid5(uuid.UUID(scope['id']), 'decode_info_json')),
                                                        name='decode_info_json', type='STRING', linkIds=[lid], pos=[1180,60+slot*20]))
                            rid = max(l[0] for l in data['links']) + 1
                            instance['outputs'].append(dict(name='decode_info_json',type='STRING',links=[rid]))
                            movie['type'] = 'MMH3CreateVideo'
                            movie['inputs'].append(dict(name='decode_info_json',type='STRING',link=rid))
                            movie['properties']['Node name for S&R'] = movie['type']
                            data['links'].append([rid,instance['id'],slot,movie['id'],len(movie['inputs'])-1,'STRING'])
                            data['last_link_id'] = rid
                if target not in nodes:
                    continue
                movie = nodes[target]
                if movie['type'] != 'CreateVideo' or movie['inputs'][tslot]['name'] != 'images':
                    continue
                movie['type'] = 'MMH3CreateVideo'
                movie['inputs'].append(dict(name='decode_info_json', type='STRING', link=None))
                add_link(decoder['id'], 1, target, len(movie['inputs'])-1, 'STRING')
        for node in nodes.values():
            if 'Node name for S&R' in node.get('properties', {}):
                node['properties']['Node name for S&R'] = node['type']
    data['nodes'] = [n for n in data['nodes'] if not n.pop('_remove_decode_toggle', False)]
    for scope in scopes:
        for movie in scope['nodes']:
            if movie['type'] != 'MMH3CreateVideo' or any(p['name']=='color_space' for p in movie['inputs']):
                continue
            slot = next(i for i,p in enumerate(movie['inputs']) if p['name']=='decode_info_json')
            movie['inputs'].insert(slot,dict(name='color_space',type='COMBO',link=None,widget={'name':'color_space'}))
            movie['widgets_values'].append('sRGB')
            for link in scope['links']:
                if isinstance(link,dict) and link['target_id']==movie['id'] and link['target_slot']>=slot:
                    link['target_slot'] += 1
                elif isinstance(link,list) and link[3]==movie['id'] and link[4]>=slot:
                    link[4] += 1


def main():
    root = Path(__file__).resolve().parent
    owned = set()
    for path in (root/'workflow_recipes').glob('*.recipe.json'):
        data = json.loads(path.read_text(encoding='utf-8'))
        owned.update((root/p).resolve() for p in data['outputs'].values())
        before = copy.deepcopy(data)
        migrate_recipe(data)
        if data != before:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    for directory in ('example_workflows', 'automation/workflows', 'tests/fixtures/workflows', 'subgraphs'):
        for path in (root/directory).glob('*.json'):
            if path.resolve() in owned:
                continue
            data = json.loads(path.read_text(encoding='utf-8'))
            before = copy.deepcopy(data)
            if 'nodes' in data:
                migrate_ui(data)
            else:
                migrate_api(data)
            if data != before:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
