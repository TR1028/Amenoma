#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate Amenoma's game data tables from Genshin Optimizer (GO) and genshin-db.

Usage (from the repository root)::

    python ArtScanner/Tools/update_game_data.py --version 7.1          # fetch + write
    python ArtScanner/Tools/update_game_data.py --version 7.1 --check  # exit 1 if stale
    python ArtScanner/Tools/update_game_data.py --version 7.1 --offline  # cache only

Outputs (``ArtScanner/rcc/``, ``NN`` = version without the dot, e.g. ``v71``):

``artifact_sets_vNN.json``
    Every artifact set released *after* the sets hard-coded in ``ArtsInfo.py``
    (good / mona / mingyu keys, CHS + EN set and piece names, rarities, slots), in GO
    release order.  ``ArtsInfo`` appends them to its positional lists, so this list is
    APPEND-ONLY: entries of the previous ``artifact_sets_v*.json`` keep their position and
    new sets are only ever added at the end (set ids are list indices).
``char_map_vNN.json``
    ``{"EN": {display name: GOOD location key}, "CHS": {...}}``.  Both travelers map to the
    GOOD location key ``"Traveler"`` (GO silently drops unknown keys such as "Aether").
``material_names_vNN.json``
    ``{"EN": {display name: GOOD material key}, "CHS": {...}}``.  All entries of the base
    file are kept verbatim; new inventory materials (genshin-db ``materials`` folder, matched
    to GO keys via the English name) are added.

Python 3.8 compatible, standard library only.  Downloads are cached in
``<tmp>/amenoma_data_cache`` (``--refresh`` ignores the cache, ``--offline`` never touches
the network).  Set ``GITHUB_TOKEN`` to lift the GitHub API rate limit.
"""
import argparse
import ast
import glob
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ARTSCANNER_DIR = os.path.dirname(TOOLS_DIR)
RCC_DIR = os.path.join(ARTSCANNER_DIR, 'rcc')
ARTSINFO_PY = os.path.join(ARTSCANNER_DIR, 'ArtsInfo.py')

GO_RAW = 'https://raw.githubusercontent.com/frzyc/genshin-optimizer/{ref}/'
GO_ALLSTAT = 'libs/gi/stats/src/allStat_gen.json'
GO_CHAR_CONSTS = 'libs/gi/consts/src/character.ts'
GO_LOCALE = 'libs/gi/dm-localization/assets/locales/{lang}/{name}_gen.json'
GDB_MATERIAL_LIST = ('https://api.github.com/repos/theBowja/genshin-db/contents/'
                     'src/data/English/materials?ref={ref}')
GDB_PACKAGE_JSON = 'https://raw.githubusercontent.com/theBowja/genshin-db/{ref}/package.json'

SLOTS = ['flower', 'plume', 'sands', 'goblet', 'circlet']
TRAVELER_LOCATION_KEY = 'Traveler'
LANGS = {'CHS': 'chs', 'EN': 'en'}


class DataError(Exception):
    pass


# --------------------------------------------------------------------------- fetching
class Fetcher(object):
    def __init__(self, cache_dir, offline=False, refresh=False, verbose=True):
        self.cache_dir = cache_dir
        self.offline = offline
        self.refresh = refresh
        self.verbose = verbose
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)

    def _cache_path(self, url):
        digest = hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]
        base = re.sub(r'[^A-Za-z0-9_.-]', '_', url.rsplit('/', 1)[-1])[:80]
        return os.path.join(self.cache_dir, digest + '_' + base)

    def get(self, url):
        path = self._cache_path(url)
        if os.path.exists(path) and not self.refresh:
            with open(path, 'rb') as f:
                return f.read()
        if self.offline:
            raise DataError('not in cache (offline mode): ' + url)
        if self.verbose:
            print('  GET ' + url, file=sys.stderr)
        req = urllib.request.Request(url, headers={'User-Agent': 'Amenoma-update-game-data'})
        token = os.environ.get('GITHUB_TOKEN')
        if token and 'api.github.com' in url:
            req.add_header('Authorization', 'token ' + token)
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            raise DataError('HTTP %s for %s' % (e.code, url))
        except urllib.error.URLError as e:
            raise DataError('network error for %s: %s' % (url, e))
        tmp = path + '.part'
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)
        return data

    def json(self, url):
        return json.loads(self.get(url).decode('utf-8'))

    def text(self, url):
        return self.get(url).decode('utf-8')


# --------------------------------------------------------------------------- helpers
def version_suffix(version):
    """'7.1' / '7.1.0' -> ('v71', '7.1.0')."""
    parts = [p for p in str(version).strip().lstrip('vV').split('.') if p != '']
    if len(parts) < 2 or not all(p.isdigit() for p in parts):
        raise DataError('bad --version %r (expected e.g. 7.1)' % (version,))
    full = '.'.join((parts + ['0'])[:3])
    return 'v' + parts[0] + parts[1], full


def normalize_name(s):
    """Normalise display names the way the OCR sees them (single spaces, ASCII quotes)."""
    s = (s or '').replace(u'\u2019', "'").replace(u'\u2018', "'").replace(u'\u00a0', ' ')
    s = s.replace(u'\u201c', '"').replace(u'\u201d', '"')
    return re.sub(r'\s+', ' ', s).strip()


def mingyu_key(good_key):
    """Best-effort MingyuLab / Genmo key: snake_case of the GOOD key."""
    return re.sub(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', '_', good_key).lower()


def norm_match(s):
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())


def has_cjk(s):
    return re.search(u'[\u3400-\u9fff\uf900-\ufaff]', s or '') is not None


def literal_list_from_py(path, name):
    """Read a literal list assigned at module level in ``path`` without importing it."""
    with io.open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise DataError('%s not found in %s' % (name, path))


def parse_ts_string_array(ts_source, const_name):
    m = re.search(r'export const %s\s*=\s*\[(.*?)\]\s*as const' % re.escape(const_name),
                  ts_source, re.S)
    if not m:
        raise DataError('could not find %s in GO character.ts' % const_name)
    return re.findall(r"'([A-Za-z0-9]+)'", m.group(1))


def load_json_file(path):
    with io.open(path, encoding='utf-8') as f:
        return json.load(f)


def dump_json_text(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + '\n'


def latest_file(pattern, exclude=None):
    def key(p):
        m = re.search(r'_v(\d+)\.json$', p)
        return int(m.group(1)) if m else -1
    files = [p for p in glob.glob(os.path.join(RCC_DIR, pattern)) if p != exclude]
    files.sort(key=key)
    return files[-1] if files else None


# --------------------------------------------------------------------------- builders
def build_artifact_sets(fetch, go_ref, suffix, full_version, prev_path):
    allstat = fetch.json(GO_RAW.format(ref=go_ref) + GO_ALLSTAT)
    art_data = allstat['art']['data']  # dict in GO release order
    release_order = list(art_data.keys())

    base_keys = literal_list_from_py(ARTSINFO_PY, 'SetNamesGOOD')
    previous = []
    if prev_path and os.path.exists(prev_path):
        previous = [s['good'] for s in load_json_file(prev_path).get('sets', [])]

    unknown_base = [k for k in base_keys if k not in art_data]
    if unknown_base:
        raise DataError('ArtsInfo.SetNamesGOOD keys unknown to GO: %s' % unknown_base)

    ordered = [k for k in previous if k not in base_keys]
    ordered += [k for k in release_order if k not in base_keys and k not in ordered]

    def locale(lang, name):
        return fetch.json(GO_RAW.format(ref=go_ref) + GO_LOCALE.format(lang=lang, name=name))

    set_names = {lang: locale(code, 'artifactNames') for lang, code in LANGS.items()}
    sets, warnings = [], []
    for key in ordered:
        if key not in art_data:
            warnings.append('set %s vanished from GO data; kept from previous file' % key)
            old = [s for s in load_json_file(prev_path)['sets'] if s['good'] == key][0]
            sets.append(old)
            continue
        info = art_data[key]
        entry = {
            'good': key,
            'mona': key,
            'mingyu': mingyu_key(key),
            'name_chs': None,
            'name_en': None,
            'rarities': sorted(info.get('rarities', [])),
            'slots': [s for s in SLOTS if s in info.get('slots', SLOTS)],
            'pieces_chs': {},
            'pieces_en': {},
            'release_index': release_order.index(key),
        }
        for lang, code in LANGS.items():
            data = locale(code, 'artifact_%s' % key)
            set_name = normalize_name(data.get('setName') or set_names[lang].get(key, ''))
            if not set_name:
                raise DataError('no %s set name for %s' % (lang, key))
            entry['name_' + code] = set_name
            pieces = data.get('pieces', {})
            for slot in entry['slots']:
                piece = normalize_name((pieces.get(slot) or {}).get('name', ''))
                if not piece:
                    raise DataError('no %s %s piece name for %s' % (lang, slot, key))
                entry['pieces_' + code][slot] = piece
        sets.append(entry)

    rarities = {k: sorted(v.get('rarities', [])) for k, v in art_data.items()}
    slots = {k: [s for s in SLOTS if s in v.get('slots', SLOTS)] for k, v in art_data.items()}
    out = {
        'target_version': full_version,
        'generator': 'ArtScanner/Tools/update_game_data.py',
        'sources': {'genshin-optimizer': go_ref},
        'note': ('Append-only: ArtsInfo appends these sets (in this order) after its '
                 'hard-coded lists; set ids are list indices.'),
        'base_sets': len(base_keys),
        'sets': sets,
        'rarities': rarities,
        'slots': slots,
    }
    return out, warnings


def build_char_map(fetch, go_ref, base_path):
    raw = GO_RAW.format(ref=go_ref)
    keys = parse_ts_string_array(fetch.text(raw + GO_CHAR_CONSTS), 'nonTravelerCharacterKeys')
    names = {lang: fetch.json(raw + GO_LOCALE.format(lang=code, name='charNames'))
             for lang, code in LANGS.items()}
    out, warnings = {'EN': {}, 'CHS': {}}, []
    for lang in ('EN', 'CHS'):
        table = out[lang]
        for key in keys:
            name = normalize_name(names[lang].get(key, ''))
            if not name:
                warnings.append('%s: no name for %s' % (lang, key))
                continue
            if lang == 'CHS' and not has_cjk(name):
                warnings.append('CHS: skipped untranslated placeholder name %r for %s' % (name, key))
                continue
            if name in table and table[name] != key:
                warnings.append('%s: duplicate name %s (%s / %s)' % (lang, name, table[name], key))
            table[name] = key
        for tkey in ('TravelerM', 'TravelerF'):
            name = normalize_name(names[lang].get(tkey, ''))
            if name:
                table[name] = TRAVELER_LOCATION_KEY
    valid = set(keys) | {TRAVELER_LOCATION_KEY}
    if base_path and os.path.exists(base_path):
        base = load_json_file(base_path)
        for lang in ('EN', 'CHS'):
            for name, key in base.get(lang, {}).items():
                if key in ('Aether', 'Lumine'):
                    key = TRAVELER_LOCATION_KEY
                if name not in out[lang]:
                    if key in valid:
                        out[lang][name] = key  # keep legacy spelling as an alias
                        warnings.append('%s: kept legacy alias %s -> %s' % (lang, name, key))
                    else:
                        warnings.append('%s: dropped legacy entry %s -> %s' % (lang, name, key))
    for lang in out:
        out[lang] = dict(sorted(out[lang].items(), key=lambda kv: (kv[1], kv[0])))
    return out, warnings


def build_material_names(fetch, go_ref, gdb_ref, base_path):
    raw = GO_RAW.format(ref=go_ref)
    go_names = {lang: fetch.json(raw + GO_LOCALE.format(lang=code, name='material'))
                for lang, code in LANGS.items()}
    listing = fetch.json(GDB_MATERIAL_LIST.format(ref=gdb_ref))
    if not isinstance(listing, list):
        raise DataError(('unexpected genshin-db listing: %r' % (listing,))[:300])
    warnings = []
    if len(listing) >= 1000:
        warnings.append('genshin-db listing has %d entries: the GitHub contents API caps '
                        'directories at 1000, some materials may be missing' % len(listing))
    slugs = sorted(x['name'][:-5] for x in listing if x.get('name', '').endswith('.json'))

    by_norm = {}
    for key, value in go_names['EN'].items():
        n = norm_match(value.get('name', ''))
        if n:
            by_norm.setdefault(n, []).append(key)

    base = {'EN': {}, 'CHS': {}}
    if base_path and os.path.exists(base_path):
        base = load_json_file(base_path)
    out = {'EN': dict(base.get('EN', {})), 'CHS': dict(base.get('CHS', {}))}
    known_keys = set(out['EN'].values()) | set(out['CHS'].values())

    unmatched, added = [], []
    for slug in slugs:
        cands = by_norm.get(slug, [])
        if len(cands) != 1:
            unmatched.append(slug if not cands else '%s (ambiguous: %s)' % (slug, cands))
            continue
        key = cands[0]
        if key in known_keys:
            continue
        en = normalize_name(go_names['EN'][key].get('name', ''))
        chs = normalize_name(go_names['CHS'].get(key, {}).get('name', ''))
        if not en or not chs:
            unmatched.append('%s (missing localisation)' % slug)
            continue
        if not has_cjk(chs):
            unmatched.append('%s (untranslated CHS name %r)' % (slug, chs))
            continue
        clash = [(lang, name) for lang, name in (('EN', en), ('CHS', chs))
                 if name in out[lang] and out[lang][name] != key]
        if clash:
            warnings.append('material %s skipped, name clash %s' % (key, clash))
            continue
        out['EN'][en] = key
        out['CHS'][chs] = key
        known_keys.add(key)
        added.append(key)
    if unmatched:
        warnings.append('%d genshin-db materials without a unique GO key (skipped): %s'
                        % (len(unmatched), ', '.join(unmatched)))
    for lang in out:
        out[lang] = dict(sorted(out[lang].items(), key=lambda kv: (kv[1], kv[0])))
    return out, warnings, added


# --------------------------------------------------------------------------- main
def diff_maps(old, new):
    lines = []
    for lang in ('EN', 'CHS'):
        o, n = old.get(lang, {}), new.get(lang, {})
        added = sorted(set(n) - set(o))
        removed = sorted(set(o) - set(n))
        changed = sorted(k for k in set(o) & set(n) if o[k] != n[k])
        lines.append('  %s: +%d -%d ~%d' % (lang, len(added), len(removed), len(changed)))
        for k in changed:
            lines.append('    ~ %s: %s -> %s' % (k, o[k], n[k]))
        for k in removed:
            lines.append('    - %s: %s' % (k, o[k]))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--version', required=True, help='game version, e.g. 7.1')
    ap.add_argument('--go-ref', default='master', help='genshin-optimizer git ref')
    ap.add_argument('--gdb-ref', default='main', help='genshin-db git ref')
    ap.add_argument('--base', default='v34',
                    help='suffix of the char_map/material_names files to extend (default v34)')
    ap.add_argument('--out-dir', default=RCC_DIR)
    ap.add_argument('--cache-dir', default=os.path.join(tempfile.gettempdir(), 'amenoma_data_cache'))
    ap.add_argument('--offline', action='store_true', help='use cached downloads only')
    ap.add_argument('--refresh', action='store_true', help='ignore cached downloads')
    ap.add_argument('--check', action='store_true',
                    help='do not write; exit 1 if the generated files differ from disk')
    args = ap.parse_args(argv)

    suffix, full_version = version_suffix(args.version)
    fetch = Fetcher(args.cache_dir, offline=args.offline, refresh=args.refresh)
    out_paths = {
        'artifact_sets': os.path.join(args.out_dir, 'artifact_sets_%s.json' % suffix),
        'char_map': os.path.join(args.out_dir, 'char_map_%s.json' % suffix),
        'material_names': os.path.join(args.out_dir, 'material_names_%s.json' % suffix),
    }
    try:
        try:
            pkg = fetch.json(GDB_PACKAGE_JSON.format(ref=args.gdb_ref))
            print('genshin-db: %s' % pkg.get('description', '?'), file=sys.stderr)
        except DataError as e:
            print('warning: genshin-db package.json unavailable: %s' % e, file=sys.stderr)

        prev_sets = out_paths['artifact_sets'] if os.path.exists(out_paths['artifact_sets']) \
            else latest_file('artifact_sets_v*.json')
        sets, w_sets = build_artifact_sets(fetch, args.go_ref, suffix, full_version, prev_sets)
        base_chars = os.path.join(RCC_DIR, 'char_map_%s.json' % args.base)
        chars, w_chars = build_char_map(fetch, args.go_ref, base_chars)
        base_mats = os.path.join(RCC_DIR, 'material_names_%s.json' % args.base)
        mats, w_mats, added_mats = build_material_names(fetch, args.go_ref, args.gdb_ref, base_mats)
    except DataError as e:
        print('error: %s' % e, file=sys.stderr)
        return 2

    for w in w_sets + w_chars + w_mats:
        print('warning: ' + w, file=sys.stderr)
    print('artifact sets: %d base (ArtsInfo) + %d generated: %s' % (
        sets['base_sets'], len(sets['sets']), ', '.join(s['good'] for s in sets['sets'])))
    print('characters: EN %d / CHS %d names' % (len(chars['EN']), len(chars['CHS'])))
    for line in diff_maps(load_json_file(base_chars) if os.path.exists(base_chars) else {}, chars):
        print(line)
    print('materials: EN %d / CHS %d names (+%d new keys vs %s)' % (
        len(mats['EN']), len(mats['CHS']), len(added_mats), args.base))

    payloads = {'artifact_sets': sets, 'char_map': chars, 'material_names': mats}
    stale = []
    for name, obj in payloads.items():
        path = out_paths[name]
        text = dump_json_text(obj)
        current = None
        if os.path.exists(path):
            with io.open(path, encoding='utf-8') as f:
                current = f.read()
        if current == text:
            print('up to date: %s' % os.path.relpath(path))
            continue
        stale.append(path)
        if args.check:
            print('STALE: %s' % os.path.relpath(path))
        else:
            with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(text)
            print('wrote: %s' % os.path.relpath(path))
    if args.check and stale:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
