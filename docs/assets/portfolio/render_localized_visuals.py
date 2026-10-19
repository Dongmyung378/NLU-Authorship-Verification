"""Render localized SVGs without changing retained source figures or data marks.

Uses Python's standard library, localization.json and the declared source SVGs.
Data geometry, colors and numerical text values remain unchanged. The declared
conceptual-thumbnail exception replaces its illustrative connector to include
both inputs; measured charts have no geometry exceptions. No model, dataset,
measurement report, screenshot or network resource is executed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
NUMBER = re.compile(r'\d+(?:,\d{3})*(?:\.\d+)?')
SHAPES = {'rect', 'line', 'path', 'circle', 'polygon', 'polyline', 'ellipse'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Output and source paths must stay within their roots')
    return path


def clarify_paired_inputs(svg):
    """Replace only the thumbnail's illustrative arrow, never data marks."""
    old = [element for element in svg if (
        element.tag == f'{{{NS}}}line'
        and all(element.get(key) == value for key, value in
                {'x1': '783', 'y1': '333', 'x2': '827', 'y2': '333'}.items())
    ) or (element.tag == f'{{{NS}}}path' and element.get('d') ==
          'M811.692,340.394 L827.000,333.000 L811.692,325.606')]
    if len(old) != 2:
        raise ValueError('The retained illustrative arrow does not match its declared source')
    baseline = [(el.tag, el.attrib.copy()) for el in svg.iter()
                if el.tag.rsplit('}', 1)[-1] in SHAPES and el not in old]
    for element in old:
        svg.remove(element)
    common = {'fill': 'none', 'stroke': '#7D3548', 'stroke-width': '4',
              'stroke-linejoin': 'round', 'data-illustrative-connector': 'true'}
    ET.SubElement(svg, f'{{{NS}}}path', {**common, 'd': 'M222 478V500H802V333H838',
                                         'data-connection': 'text-a-to-model-families'})
    ET.SubElement(svg, f'{{{NS}}}path', {**common, 'd': 'M601 478V500',
                                         'data-connection': 'text-b-to-pair-junction'})
    ET.SubElement(svg, f'{{{NS}}}circle', {'cx': '601', 'cy': '500', 'r': '5',
                                          'fill': '#7D3548', 'data-illustrative-connector': 'true'})
    ET.SubElement(svg, f'{{{NS}}}path', {**common, 'd': 'M825 325L838 333L825 341',
                                         'stroke-linecap': 'round'})
    preserved = [(el.tag, el.attrib.copy()) for el in svg.iter()
                 if el.tag.rsplit('}', 1)[-1] in SHAPES
                 and el.get('data-illustrative-connector') != 'true']
    if preserved != baseline:
        raise ValueError('An illustrative refinement changed retained data geometry or palette')


def render(spec):
    source = inside(REPOSITORY, spec['source'])
    raw = source.read_bytes()
    svg = ET.fromstring(raw)
    changed = 0
    for element in svg.iter():
        if element.tag.rsplit('}', 1)[-1] not in ('text', 'title', 'desc'):
            continue
        original = element.text or ''
        translation = spec['translations'].get(original)
        if translation is None:
            if spec['language'] == 'en' and re.search('[가-힣]', original):
                raise ValueError(f'Missing English translation: {original}')
            continue
        options = {'text': translation} if isinstance(translation, str) else translation
        text = options['text']
        if Counter(NUMBER.findall(original)) != Counter(NUMBER.findall(text)):
            raise ValueError(f'Translation changed numerical tokens: {original!r}')
        element.text = text
        for key, value in options.items():
            if key not in ('text', 'font-size', 'x', 'y', 'text-anchor'):
                raise ValueError(f'Unapproved presentation option: {key}')
            if key != 'text':
                element.set(key, str(value))
        changed += 1
    refined = spec.get('clarify_paired_inputs', False)
    if refined:
        if source.name != 'nlu-thumbnail.svg':
            raise ValueError('The paired-input exception is restricted to the conceptual thumbnail')
        clarify_paired_inputs(svg)
    metadata = svg.find(f'{{{NS}}}metadata')
    if metadata is None:
        metadata = ET.SubElement(svg, f'{{{NS}}}metadata')
        record = {}
    else:
        record = json.loads(metadata.text)
    record['localization'] = {'language': spec['language'], 'source_svg': spec['source'],
                              'source_sha256': sha(raw), 'measurement_values_changed': False,
                              'illustrative_connector_refined': refined, 'data_geometry_changed': False}
    metadata.text = json.dumps(record, ensure_ascii=False, sort_keys=True)
    rendered = (ET.tostring(svg, encoding='unicode') + '\n').encode('utf-8')
    if spec['language'] == 'en' and any(re.search('[가-힣]', el.text or '') for el in svg.iter()
                                       if el.tag.rsplit('}', 1)[-1] in ('text', 'title', 'desc')):
        raise ValueError('Korean display text remains in an English figure')
    return rendered, {'source': spec['source'], 'target': spec['target'],
                      'language': spec['language'], 'source_sha256': sha(raw),
                      'sha256': sha(rendered), 'translated_elements': changed,
                      'illustrative_connector_refined': refined, 'data_geometry_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=REPOSITORY)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    config = json.loads((HERE / 'localization.json').read_text(encoding='utf-8'))
    records = []
    for spec in config['figures']:
        rendered, record = render(spec)
        target = inside(args.output_root, spec['target'])
        if target == inside(REPOSITORY, spec['source']):
            raise ValueError('Retained source figures cannot be overwritten')
        if args.check:
            if not target.exists() or target.read_bytes() != rendered:
                raise ValueError(f'Localized derivative differs: {target}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(rendered)
        records.append(record)
    receipt = {'schema_version': 1, 'presentation_only': True,
               'models_or_evaluation_run': False, 'files': records}
    path = inside(args.output_root, 'docs/assets/portfolio/localization-receipt.json')
    encoded = (json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if args.check:
        if not path.exists() or path.read_bytes() != encoded:
            raise ValueError('Localization receipt differs')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    print(json.dumps({'verified' if args.check else 'generated': len(records),
                      'receipt': str(path), 'measurements_changed': False}))


if __name__ == '__main__':
    main()
