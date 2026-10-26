"""Fixed physical colour limits, paired ordinal errors and reviewed differences."""
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion
from src.common import read_json, write_json, digest_file
from src.datasets.cleargrasp import read_rgb, path_for
from src.cleargrasp_paired import read_pair, contained, opaque_folder
from src.evaluation.metrics import valid_pixels
from src.evaluation.relative_geometry import oracle_scale
from src.evaluation.paired_geometry import shared_ordinal_pairs
from src.evaluation.cleargrasp_paired import (analysis_identity, protocol, differences,
    output_guard, finish_artifacts, LABEL)


def ordinal_error_maps(t, o, gt, mask, common, **params):
    pairs = shared_ordinal_pairs(gt, mask, common, **params)
    endpoints = np.concatenate(list(pairs.values()), axis=0)
    maps = []
    for pred in (t, o):
        count = np.zeros(gt.size)
        wrong = np.zeros(gt.size)
        if len(endpoints):
            x, y = endpoints.T
            error = (np.sign(pred.ravel()[x]-pred.ravel()[y]) != np.sign(gt.ravel()[x]-gt.ravel()[y]))
            for column in (x, y):
                np.add.at(count, column, 1)
                np.add.at(wrong, column, error)
        result = np.full(gt.size, np.nan)
        np.divide(wrong, count, out=result, where=count > 0)
        maps.append(result.reshape(gt.shape))
    return maps


def render(a, rec, arrays, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t, o, gt, mask = arrays
    common = valid_pixels(t, gt) & valid_pixels(o, gt)
    bg = common & ~mask
    target.mkdir(parents=True, exist_ok=True)
    def panel(name, items, cols=2):
        rows = (len(items)+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)
        for ax, item in zip(axes.ravel(), items):
            title, data, cmap, lo, hi = item
            ax.set_title(title, fontsize=10)
            if data is None:
                ax.text(.5, .5, 'SKIPPED: '+rec['registration']['reason'], ha='center', va='center', wrap=True)
            else:
                artist = ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi)
                if cmap:
                    fig.colorbar(artist, ax=ax, fraction=.046)
            ax.axis('off')
        for ax in axes.ravel()[len(items):]:
            ax.axis('off')
        fig.suptitle(rec['pair_id'])
        fig.tight_layout()
        fig.savefig(target/name, dpi=a.figure_dpi)
        plt.close(fig)
    rgbt = read_rgb(path_for(a.data_root, rec['sample'], 'rgb'))
    rgbo = read_rgb(contained(a.data_root, rec['opaque_rgb']))
    panel('rgb_comparison.png', [('Transparent RGB', rgbt, None, None, None),
                               ('Opaque RGB', rgbo, None, None, None)])
    panel('depth_comparison.png', [('Opaque-capture GT (m)', np.where(np.isfinite(gt) & (gt > 0), gt, np.nan),
            'viridis', 0, a.depth_max), ('Original physical object mask', mask, 'gray', 0, 1),
        ('Transparent raw metric (m)', np.where(t > 0, t, np.nan), 'viridis', 0, a.depth_max),
        ('Opaque raw metric (m)', np.where(o > 0, o, np.nan), 'viridis', 0, a.depth_max)])
    panel('error_comparison.png', [(name+' raw absolute error (m)',
        np.where(common, np.abs(pred-gt), np.nan), 'magma', 0, a.error_max)
        for name, pred in (('Transparent', t), ('Opaque', o))])
    aligned = []
    for name, pred in (('Transparent', t), ('Opaque', o)):
        scale = oracle_scale(pred, gt, bg)
        aligned.append((name+' background aligned (m)\nGT-dependent diagnostic only',
            pred*scale if scale is not None else np.full(gt.shape, np.nan), 'viridis', 0, a.depth_max))
    panel('scale_comparison.png', aligned)
    maps = ordinal_error_maps(t, o, gt, mask, common, **protocol(a))
    panel('geometry_comparison.png', [(name+' sampled ordinal endpoint error fraction', m, 'magma', 0, 1)
        for name, m in zip(('Transparent', 'Opaque'), maps)])
    d = differences(t, o, common, bg, rec['registration']['verified'])
    boundary = binary_dilation(mask, iterations=max(1, a.erosion_margin)) ^ binary_erosion(
        mask, iterations=max(1, a.erosion_margin))
    items = []
    for title, key, limit in (('T - O raw depth (m)', 'depth', a.difference_max),
                             ('log(T) - log(O)', 'log_depth', a.log_difference_max),
                             ('Background-referenced log difference', 'background_referenced_log', a.log_difference_max)):
        items.append((title, d.get(key), 'coolwarm', -limit, limit))
    items.append(('Boundary band (original mask preserved)', boundary, 'gray', 0, 1))
    panel('prediction_difference.png', items)
    region_stats = {}
    if d['status'] == 'available':
        for name, area in dict(object=mask, background=~mask, boundary=boundary).items():
            v = common & area
            region_stats[name] = dict(pixels=int(v.sum()), **{key: float(np.median(d[key][v])) if v.any() else None
                for key in ('depth', 'log_depth', 'background_referenced_log')})
    write_json(target/'visualization.json', dict(pair_id=rec['pair_id'], depth_range_metres=[0,a.depth_max],
        error_range_metres=[0,a.error_max], independent_percentile_normalization=False,
        per_pixel_difference_status=d['status'], registration=rec['registration'], difference_regions=region_stats,
        diagnostic_label=LABEL, ordinal_protocol=protocol(a), common_valid_pixels=int(common.sum())))


def visualize(a):
    manifest = analysis_identity(a)
    pairs, failures = [], []
    for rec in manifest['records']:
        try:
            if rec['status'] != 'ready':
                raise ValueError(rec['reason'])
            pairs.append((rec, read_pair(a, rec)))
        except Exception as exc:
            failures.append(dict(pair_id=rec['pair_id'], reason=str(exc)))
    signature = dict(manifest_sha256=digest_file(a.output_dir/'paired_manifest.json'),
        protocol=protocol(a), settings={k:getattr(a,k) for k in ('depth_max','error_max','difference_max',
            'log_difference_max','figure_dpi','erosion_margin')}, failures=failures,
        predictions={r['pair_id']:digest_file(opaque_folder(a.output_dir,r['sample'])/'opaque_prediction.json') for r,_ in pairs})
    if a.dry_run:
        print(f'Visualization preflight: {len(pairs)} pairs; failures: {failures}')
        return int(bool(failures))
    target = a.output_dir/'figures'
    if not output_guard(a, target, signature):
        if digest_file(a.output_dir/'contact_sheet.png') != read_json(
                target/'contact_sheet_metadata.json')['sha256']:
            raise ValueError('Contact sheet checksum mismatch')
        return int(bool(failures))
    if (a.output_dir/'contact_sheet.png').exists():
        raise FileExistsError('Contact sheet already exists')
    thumbs = []
    for rec, arrays in pairs:
        folder = contained(target, rec['pair_id'])
        try:
            render(a, rec, arrays, folder)
            with Image.open(folder/'depth_comparison.png') as im:
                thumb = ImageOps.contain(im.convert('RGB'), (640, 440))
            tile = Image.new('RGB',(640,470),'white')
            tile.paste(thumb,(0,30))
            ImageDraw.Draw(tile).text((10,8), rec['pair_id'], fill='black')
            thumbs.append(tile)
        except Exception as exc:
            failures.append(dict(pair_id=rec['pair_id'], reason=str(exc)))
    sheet = Image.new('RGB', (640*4, 470*max(1,(len(thumbs)+3)//4)), '#dddddd')
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i%4)*640,(i//4)*470))
    sheet.save(a.output_dir/'contact_sheet.png')
    write_json(target/'contact_sheet_metadata.json', dict(sha256=digest_file(a.output_dir/'contact_sheet.png')))
    write_json(target/'visualization_summary.json', dict(selected=len(manifest['records']), rendered=len(thumbs),
        failures=failures, pixel_differences_skipped=sum(not r['registration']['verified'] for r,_ in pairs)))
    finish_artifacts(target, signature)
    return int(bool(failures))
