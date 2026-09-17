"""Explicit migration of paired video/archive outputs; preserves node IDs."""
from __future__ import annotations

import json
import uuid
from pathlib import Path


def migrate_ui(data):
    videos = [n for n in data.get('nodes', []) if n['type'] == 'SaveVideo']
    archives = [n for n in data.get('nodes', []) if n['type'] == 'MMH3Save']
    if len(videos) != 1 or len(archives) != 1:
        return False
    video, archive = videos[0], archives[0]
    links = data['links']
    incoming = next(l for l in links if l[3:5] == [archive['id'], 0])
    old_source, old_slot = incoming[1:3]
    values = video['widgets_values']
    if len(values) != 3:
        raise ValueError('Review non-default codec controls before migrating')
    # Existing VIDEO consumers expect the original value; do not silently change
    # those consumers to a lossy file-backed value.
    if any(o.get('links') for o in video.get('outputs', [])):
        raise ValueError('SaveVideo already has consumers; review manually')
    video['type'] = 'MMH3SaveVideo'
    video.setdefault('properties', {})['Node name for S&R'] = video['type']
    video['title'] = 'Save Video · encode once'
    video['inputs'] = [video['inputs'][0]] + [
        {'name': name, 'type': kind, 'link': None, 'widget': {'name': name}}
        for name, kind in [('filename_prefix', 'STRING'), ('format', 'COMBO'), ('codec', 'COMBO')]]
    new_id = max([data.get('last_link_id', 0)] + [l[0] for l in links]) + 1
    video['inputs'].append({'name': 'packet', 'type': 'MMH3_MEDIA', 'link': new_id})
    video['outputs'] = [{'name': 'video', 'type': 'VIDEO', 'links': []},
                        {'name': 'packet', 'type': 'MMH3_MEDIA', 'links': [incoming[0]]}]
    source_node = next(n for n in data['nodes'] if n['id'] == old_source)
    source_links = source_node['outputs'][old_slot]['links']
    source_links[source_links.index(incoming[0])] = new_id
    links.append([new_id, old_source, old_slot, video['id'], 4, 'MMH3_MEDIA'])
    incoming[1:3] = [video['id'], 1]
    data['last_link_id'] = new_id
    return True


def migrate_api(data):
    videos = [(k,n) for k,n in data.items() if isinstance(n,dict) and n.get('class_type') == 'SaveVideo']
    archives = [n for n in data.values() if isinstance(n,dict) and n.get('class_type') == 'MMH3Save']
    if len(videos) != 1 or len(archives) != 1:
        return False
    key, node = videos[0]
    node['class_type'] = 'MMH3SaveVideo'
    node['inputs']['packet'] = archives[0]['inputs']['packet']
    archives[0]['inputs']['packet'] = [key, 1]
    return True


def migrate_recipe(data):
    nodes = data['graph']['nodes']
    videos = [(k,n) for k,n in nodes.items() if n['type'] == 'SaveVideo']
    archives = [n for n in nodes.values() if n['type'] == 'MMH3Save']
    if len(videos) != 1 or len(archives) != 1:
        return False
    key, node = videos[0]
    node['type'] = 'MMH3SaveVideo'
    values = node['values']
    values['codec'] = values.pop('format.codec', values.get('codec', 'auto'))
    node['bindings']['packet'] = archives[0]['bindings']['packet']
    archives[0]['bindings']['packet'] = [key, 'packet']
    node['layout']['title'] = 'Save Video · encode once'
    node['layout'].setdefault('properties', {})['Node name for S&R'] = node['type']
    return True


def add_f01_tae_switch(data):
    sub = next(s for s in data['definitions']['subgraphs'] if s['name'] == 'H3 Sampling + AV Decode')
    existing = next((n for n in sub['nodes'] if n['type'] == 'MMH3H3VideoDecode'), None)
    if existing is not None:
        if any(p['name'] == 'decode_mode' for p in existing['inputs']):
            return False
        if existing['inputs'][2]['name'] == 'draft_tae':
            return False
        existing['inputs'] = existing['inputs'][:3]
        existing['inputs'][2]['name'] = 'draft_tae'
        existing['inputs'][2]['widget'] = {'name': 'draft_tae'}
        existing['widgets_values'] = [False]
        instance = next(n for n in data['nodes'] if n['type'] == sub['id'])
        for port in sub['inputs'] + instance['inputs']:
            if port['name'] == 'fast_vae':
                port['name'] = 'draft_tae'
        for node in data['nodes']:
            if node.get('title') == 'Fast VAE':
                node['title'] = 'Draft TAE'
                node['widgets_values'] = [False]
        return True
    decode = next(n for n in sub['nodes'] if n['type'] == 'VAEDecode')
    decode['type'] = 'MMH3H3VideoDecode'
    decode['properties']['Node name for S&R'] = decode['type']
    decode['inputs'].extend([
        {'name': 'draft_tae', 'type': 'BOOLEAN', 'link': None, 'widget': {'name': 'draft_tae'}}])
    decode['widgets_values'] = [False]
    decode['size'][1] = max(decode['size'][1], 180)
    slot = len(sub['inputs'])
    link_id = max(l['id'] for l in sub['links']) + 1
    sub['inputs'].append({'id': str(uuid.uuid5(uuid.UUID(sub['id']), 'draft_tae')), 'name': 'draft_tae',
                          'type': 'BOOLEAN', 'linkIds': [link_id], 'pos': [-180, 40 + slot*20]})
    sub['links'].append({'id':link_id, 'origin_id':-10, 'origin_slot':slot,
                        'target_id':decode['id'], 'target_slot':2, 'type':'BOOLEAN'})
    decode['inputs'][2]['link'] = link_id
    sub['state']['lastLinkId'] = link_id
    sub['inputNode']['bounding'][3] += 20
    instance = next(n for n in data['nodes'] if n['type'] == sub['id'])
    node_id = max(n['id'] for n in data['nodes']) + 1
    root_link = max(l[0] for l in data['links']) + 1
    instance['inputs'].append({'name':'draft_tae', 'type':'BOOLEAN', 'link':root_link})
    instance['size'][1] += 24
    data['nodes'].append({'id':node_id, 'type':'PrimitiveBoolean', 'title':'Draft TAE',
        'pos':[instance['pos'][0], instance['pos'][1] + instance['size'][1] + 50],
        'size':[300,100], 'flags':{}, 'order':instance['order'], 'mode':0,
        'inputs':[{'name':'value','type':'BOOLEAN','link':None,'widget':{'name':'value'}}],
        'outputs':[{'name':'BOOLEAN','type':'BOOLEAN','links':[root_link]}],
        'properties':{'Node name for S&R':'PrimitiveBoolean'}, 'widgets_values':[False]})
    data['links'].append([root_link,node_id,0,instance['id'],slot,'BOOLEAN'])
    data['last_node_id'] = node_id
    data['last_link_id'] = root_link
    return True


def add_tae_switches(data):
    """Only replace image decodes feeding video assembly, including subgraph exports."""
    changed = False
    for scope in [data] + data.get('definitions', {}).get('subgraphs', []):
        nodes = {n['id']: n for n in scope['nodes']}
        selected = []
        for node in nodes.values():
            if node['type'] != 'VAEDecode':
                continue
            targets = [(l['target_id'], l['target_slot']) if isinstance(l, dict) else (l[3], l[4])
                       for l in scope['links'] if (l['origin_id'] if isinstance(l, dict) else l[1]) == node['id']]
            if scope is not data:
                resolved = []
                for target, slot in targets:
                    if target != -20:
                        resolved.append((nodes.get(target, {}).get('type'), slot))
                    else:
                        for instance in data['nodes']:
                            if instance['type'] == scope['id']:
                                resolved.extend((next(n['type'] for n in data['nodes'] if n['id'] == l[3]), l[4])
                                                for l in data['links'] if l[1:3] == [instance['id'], slot])
            else:
                resolved = [(nodes.get(target, {}).get('type'), slot) for target, slot in targets]
            if resolved and all(t == 'CreateVideo' and slot == 0 for t, slot in resolved):
                selected.append(node)
        if not selected:
            continue
        for node in selected:
            node['type'] = 'MMH3H3VideoDecode'
            node.setdefault('properties', {})['Node name for S&R'] = node['type']
            node['inputs'].append({'name': 'draft_tae', 'type': 'BOOLEAN', 'link': None, 'widget': {'name': 'draft_tae'}})
            node['widgets_values'] = [False]
            node['size'][1] = max(node['size'][1], 150)
        if scope is not data:
            slot = len(scope['inputs'])
            links = []
            for node in selected:
                lid = max(l['id'] for l in scope['links']) + 1
                scope['links'].append({'id': lid, 'origin_id': -10, 'origin_slot': slot,
                                      'target_id': node['id'], 'target_slot': 2, 'type': 'BOOLEAN'})
                node['inputs'][2]['link'] = lid
                links.append(lid)
            scope['inputs'].append({'id': str(uuid.uuid5(uuid.UUID(scope['id']), 'draft_tae')),
                                    'name': 'draft_tae', 'type': 'BOOLEAN', 'linkIds': links,
                                    'pos': [-180, 40 + slot * 20]})
            scope['state']['lastLinkId'] = max(links)
            scope['inputNode']['bounding'][3] = max(scope['inputNode']['bounding'][3], 60 + slot * 20)
            for instance in list(data['nodes']):
                if instance['type'] != scope['id']:
                    continue
                nid = max(n['id'] for n in data['nodes']) + 1
                lid = max(l[0] for l in data['links']) + 1
                instance['inputs'].append({'name': 'draft_tae', 'type': 'BOOLEAN', 'link': lid})
                instance['size'][1] += 24
                data['nodes'].append({'id': nid, 'type': 'PrimitiveBoolean', 'title': 'Draft TAE',
                    'pos': [instance['pos'][0], instance['pos'][1] + instance['size'][1] + 50],
                    'size': [300, 100], 'flags': {}, 'order': instance['order'], 'mode': 0,
                    'inputs': [{'name': 'value', 'type': 'BOOLEAN', 'link': None, 'widget': {'name': 'value'}}],
                    'outputs': [{'name': 'BOOLEAN', 'type': 'BOOLEAN', 'links': [lid]}],
                    'properties': {'Node name for S&R': 'PrimitiveBoolean'}, 'widgets_values': [False]})
                data['links'].append([lid, nid, 0, instance['id'], slot, 'BOOLEAN'])
                data['last_node_id'], data['last_link_id'] = nid, lid
        changed = True
    return changed


def add_api_tae_switches(data):
    changed = False
    for key, node in data.items():
        if not isinstance(node, dict) or node.get('class_type') != 'VAEDecode':
            continue
        consumers = [(n.get('class_type'), name) for n in data.values() if isinstance(n, dict)
                     for name, value in n.get('inputs', {}).items() if value == [key, 0]]
        if consumers and all(t == 'CreateVideo' and name == 'images' for t, name in consumers):
            node['class_type'] = 'MMH3H3VideoDecode'
            node['inputs']['draft_tae'] = False
            changed = True
    return changed


def main():
    root = Path(__file__).resolve().parent
    changed = []
    for directory in ('example_workflows', 'automation/workflows', 'tests/fixtures/workflows', 'workflow_recipes'):
        for path in sorted((root/directory).glob('*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            if path.name.endswith('.recipe.json'):
                edit = migrate_recipe(data)
            elif 'nodes' in data:
                edit = migrate_ui(data)
                edit = add_tae_switches(data) or edit
            elif directory == 'workflow_recipes':
                continue
            else:
                edit = migrate_api(data)
                edit = add_api_tae_switches(data) or edit
            if edit:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
                changed.append(str(path.relative_to(root)))
    print('\n'.join(changed))


if __name__ == '__main__':
    main()
