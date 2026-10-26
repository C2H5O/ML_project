"""Scientific figures for saved-array CPU diagnostics; no model access."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from src.datasets.cleargrasp_paths import sample_parts, full_sample_id, safe_result_path
from src.evaluation.metrics import valid_pixels


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def pair_value_map(shape, pairs, values):
    result = np.full(int(np.prod(shape)), np.nan)
    if len(pairs):
        count = np.bincount(pairs[:, 0], minlength=result.size)
        sums = np.bincount(pairs[:, 0], weights=values, minlength=result.size)
        used = count > 0
        result[used] = sums[used]/count[used]
    return result.reshape(shape)


def render_analysis_sample(output, sample, rgb, gt, pred, mask, bg_scale,
                           geometry, details, boundary, boundary_details,
                           depth_max=2., error_max=.5):
    plt = _plotting()
    destination = safe_result_path(output, 'figures', *sample_parts(sample), 'analysis.png')
    destination.parent.mkdir(parents=True, exist_ok=True)
    valid = valid_pixels(pred, gt)
    pv = np.isfinite(pred) & (pred > 0)
    gv = np.isfinite(gt) & (gt > 0)
    aligned = pred*bg_scale if bg_scale is not None else np.full(gt.shape, np.nan)
    ordinal_map = np.full(gt.shape, np.nan)
    if geometry['ordinal']['transparent_background']['status'] == 'available':
        pairs = details['ordinal_pairs']['transparent_background']
        a, b = pairs.T
        error = (np.sign(pred.ravel()[a]-pred.ravel()[b]) !=
                 np.sign(gt.ravel()[a]-gt.ravel()[b])).astype(float)
        ordinal_map = pair_value_map(gt.shape, pairs, error)
    boundary_map = np.full(gt.shape, np.nan)
    if boundary['contrast']['status'] == 'available':
        boundary_map = pair_value_map(gt.shape, boundary_details['pairs'], boundary_details['log_error'])
    mask_overlay = rgb.astype(float).copy()
    mask_overlay[mask] = .55*mask_overlay[mask] + .45*np.array([255, 0, 190])
    panels = [
        ('RGB input', rgb, None, None, None),
        ('Ground truth depth', np.where(gv, gt, np.nan), 'viridis', depth_max, 'm'),
        ('Raw metric prediction; no GT masking', np.where(pv, pred, np.nan), 'viridis', depth_max, 'm'),
        ('Background-aligned depth\nGT-Dependent Diagnostic Only', np.where(pv, aligned, np.nan), 'viridis', depth_max, 'm'),
        ('Raw absolute error', np.where(valid, np.abs(pred-gt), np.nan), 'magma', error_max, 'm'),
        ('Background-aligned error\nGT-Dependent Diagnostic Only', np.where(valid, np.abs(aligned-gt), np.nan), 'magma', error_max, 'm'),
        ('Transparent mask overlay', mask_overlay.astype(np.uint8), None, None, None),
        ('Local T/B ordinal errors at T endpoints\nGrey = unsampled / insufficient support', ordinal_map, 'magma', 1., 'error fraction'),
        ('Boundary log-contrast error at T endpoints\nGrey = unsampled / insufficient support', boundary_map, 'magma', 1., 'natural log')]
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for ax, (title, data, cmap_name, limit, unit) in zip(axes.flat, panels):
        if cmap_name is None:
            ax.imshow(data, interpolation='nearest')
        else:
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad('#999999')
            im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=limit, interpolation='nearest')
            fig.colorbar(im, ax=ax, fraction=.04, pad=.02, label=unit, extend='max')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig.suptitle(full_sample_id(sample) + ' | saved predictions, CPU analysis | fixed colour ranges')
    fig.tight_layout()
    fig.savefig(destination, dpi=110)
    plt.close(fig)
    return destination


def render_analysis_charts(output, rows, summary):
    plt = _plotting()
    folder = Path(output)/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    ids = [r['sample_id'] for r in rows]
    x = np.arange(len(rows))
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder/name, dpi=140)
        plt.close(fig)
    def labels(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=65, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=.2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ('median_pred_gt', 'oracle_scale'),
            ('Median prediction / GT (raw metric)', 'Oracle multiplicative factor\nGT-Dependent Diagnostic Only')):
        values = [r['scale'][key] for r in rows if r['scale'][key] is not None]
        ax.hist(values, bins=min(10, max(1, len(values))), color='#34788d', edgecolor='white')
        ax.axvline(1, color='black', linestyle='--', label='1.0')
        ax.set_title(title)
        ax.set_ylabel('Images')
        ax.legend()
    save(fig, 'scale_ratio_distribution.png')
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, [r['scale']['raw_absrel'] for r in rows], 'o-', label='Raw metric AbsRel')
    ax.plot(x, [r['scale']['oracle_absrel'] for r in rows], 'o-', label='Oracle aligned: GT-dependent diagnostic')
    ax.set_title('Per-image full-image error: raw versus oracle\nGT-Dependent Diagnostic Only (oracle curve)')
    ax.set_ylabel('AbsRel')
    labels(ax)
    ax.legend()
    save(fig, 'raw_vs_aligned_absrel.png')
    fig, ax = plt.subplots(figsize=(14, 6))
    for delta, key, label in [(-.18, 'transparent_background_aligned_absrel', 'Transparent'),
                               (.18, 'background_background_aligned_absrel', 'Background')]:
        vals = [r['scale'][key] if r['scale'][key] is not None else np.nan for r in rows]
        ax.bar(x+delta, vals, width=.36, label=label)
    ax.set_title('One background-derived scale per image: GT-Dependent Diagnostic Only')
    ax.set_ylabel('Residual AbsRel')
    labels(ax)
    ax.legend()
    save(fig, 'transparent_vs_background.png')
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(summary['ordinal'])
    for i, name in enumerate(names):
        item = summary['ordinal'][name]
        val = item['accuracy']
        ax.bar(i, val if val is not None else 0, color='#34788d' if val is not None else '#999999')
        ax.text(i, (val or 0)+.025, f"n={item['valid_pairs']}\nimages={item['supported_images']}" if val is not None else 'unavailable',
                ha='center', fontsize=8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Ordinal accuracy')
    ax.set_title('No GT scaling; only support-qualified per-image comparisons pooled')
    save(fig, 'ordinal_accuracy.png')
    fig, ax = plt.subplots(figsize=(14, 6))
    reversal = np.array([r['geometry']['penetration']['reversal_rate'] if
                        r['geometry']['penetration']['reversal_rate'] is not None else np.nan for r in rows])
    compression = [r['geometry']['penetration']['compression_rate'] if
                   r['geometry']['penetration']['compression_rate'] is not None else np.nan for r in rows]
    ax.bar(x, reversal, label='Order reversal')
    ax.bar(x, compression, bottom=reversal, label='Severe log-gap compression')
    for i, row in enumerate(rows):
        item = row['geometry']['penetration']
        text = f"n={item['valid_pairs']}" if item['status'] == 'available' else 'unavailable'
        ax.text(i, 1.01, text, rotation=90, fontsize=6, va='bottom', ha='center')
    ax.set_ylim(0, 1.3)
    ax.set_ylabel('Sampled pair fraction')
    ax.set_title('Local front-surface diagnostic; not proof of same-ray background penetration')
    labels(ax)
    ax.legend()
    save(fig, 'penetration_rates.png')
    selected = {}
    for group in summary['hardest_samples'].values():
        for item in group:
            selected[item['sample_id']] = item['figure']
    if selected:
        columns = min(3, len(selected))
        sheet = Image.new('RGB', (480*columns, 370*((len(selected)+columns-1)//columns)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (sid, relative) in enumerate(selected.items()):
            with Image.open(Path(output)/relative) as im:
                tile = im.convert('RGB')
                tile.thumbnail((480, 340))
            left, top = 480*(i % columns), 370*(i//columns)
            draw.text((left+5, top+5), sid, fill='black')
            sheet.paste(tile, (left, top+25))
        sheet.save(folder/'hardest_samples.png')
