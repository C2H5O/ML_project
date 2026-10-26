"""Evidence-derived course report; missing support never becomes a positive claim."""
from pathlib import Path
import numpy as np


def number(value, digits=6):
    return 'unavailable' if value is None else f'{value:.{digits}f}'


def percent(value):
    return 'unavailable' if value is None else f'{100*value:.2f}%'


def write_analysis_report(path, summary, rows):
    raw, diag = summary['raw_metric']['regions'], summary['diagnostic_metric']
    protocol, baseline = summary['protocol'], summary['baseline_comparison']
    lines = ['# ClearGrasp：尺度误差与透明物体相对几何分析', '',
        f"分析状态：**{summary['status']}**。读取保存 manifest 中的 {summary['verified_samples']}/{summary['selected_samples']} 张米制预测，",
        '全部来自已有 `metric_depth.npy`；本分析未抽取新图像、未调用模型、未修改预测或正式基线。', '',
        f"数据划分：`{protocol['split']}`。原 Raw Metric 基线对照状态：**{baseline['status']}**。", '',
        '**所有 oracle/background alignment 均为 GT-Dependent Diagnostic Only，不能作为可部署的米制性能。**', '',
        '## A. Metric Depth Performance', '',
        'Raw 指标不做 GT 对齐，按所有图像的有效像素汇总；有效区域为有限正值 GT 与预测的交集。', '',
        '| Region | AbsRel | RMSE (m) | δ1 | Valid pixels | Images |',
        '|---|---:|---:|---:|---:|---:|']
    for region, r in raw.items():
        lines.append(f"| {region} | {number(r['AbsRel'])} | {number(r['RMSE'])} | {number(r['delta1'])} | {r['valid_pixels']} | {r['samples']} |")
    if baseline['status'] == 'mismatch':
        lines.extend(['', '**重算指标与原基线不一致。正式基线文件未覆盖；应先调查以下差异，再引用实验结果。**', ''])
        if baseline.get('reason'):
            lines.append(f"- {baseline['reason']}")
        for item in baseline.get('differences', []):
            lines.append(f"- {item.get('region', '')} {item['metric']}: 原值 `{item.get('previous')}`，重算 `{item.get('recomputed')}`。")
        if baseline.get('error'):
            lines.append(f"- 原基线读取失败：{baseline['error']}")
    elif baseline['status'] == 'match':
        lines.extend(['', '重算的像素数、样本数和 Raw 指标与原文件在记录的容差内一致。'])
    else:
        lines.extend(['', '未找到原 Raw 基线文件，因此不能验证与历史结果的一致性。'])
    ratios = [r['scale']['median_pred_gt'] for r in rows if r['scale']['median_pred_gt'] is not None]
    lines.extend(['', '## B. Scale Error Decomposition', '',
        '逐图分别计算 `median(pred/GT)` 与 `median(GT/pred)`；偶数像素时二者不保证严格互为倒数。',
        '全图 oracle 只应用一个乘法因子；背景对齐因子只由非透明背景的有效 GT 求出。', '',
        f"逐图 median(pred/GT) 的中位数为 **{number(float(np.median(ratios)) if ratios else None)}**，",
        f"范围为 {number(min(ratios) if ratios else None)} 至 {number(max(ratios) if ratios else None)}。", '',
        '| Region | Raw AbsRel | Global oracle AbsRel | Background-aligned AbsRel | Oracle reduction / raw |',
        '|---|---:|---:|---:|---:|'])
    for region in raw:
        d = diag['oracle'][region]
        lines.append(f"| {region} | {number(raw[region]['AbsRel'])} | {number(d['AbsRel'])} | "
                     f"{number(diag['background_aligned'][region]['AbsRel'])} | {percent(d['raw_absrel_reduction_fraction'])} |")
    full = diag['oracle']['full_image']
    lines.extend(['', f"全图 Raw AbsRel 从 {number(raw['full_image']['AbsRel'])} 变为 GT oracle 对齐后的 "
        f"{number(full['AbsRel'])}，相对变化（减少为正）为 {percent(full['raw_absrel_reduction_fraction'])}。",
        '该变化量描述允许使用 GT 尺度时能减少多少 AbsRel，不是尺度误差的严格因果占比。',
        '这些汇总使用相同的像素加权定义；逐图尺度、log bias、SILog 和 log MAD 见 `scale_analysis.csv`。', ''])
    gaps = [r['scale']['transparent_minus_background'] for r in rows
            if r['scale']['transparent_minus_background'] is not None]
    increases = [r['scale']['transparent_relative_increase'] for r in rows
                 if r['scale']['transparent_relative_increase'] is not None]
    lines.extend([f"背景对齐后，有 {sum(v > 0 for v in gaps)}/{len(gaps)} 张可比较图像的透明区域残差高于背景；",
        f"逐图差值的中位数为 {number(float(np.median(gaps)) if gaps else None)}，",
        f"逐图相对增幅的中位数为 {percent(float(np.median(increases)) if increases else None)}。",
        '背景残差为零时相对增幅记为 unavailable。这里报告描述性差异，不宣称统计显著，也不把残差差异直接定义为穿透。', '',
        '## C. Relative Geometry', '',
        f"有效 GT 深度差严格大于 {protocol['gt_gap_m']:g} m；每类每图候选抽样 {protocol['pair_draws']} 次。",
        f"每图至少需要 {protocol['min_unique_pairs']} 个不同像素对，并且两端各至少 "
        f"{protocol['min_unique_pixels']} 个不同像素，才报告比率。低支持样本标 unavailable，且不混入汇总分母。", '',
        '| Comparison | Accuracy | Accepted pairs used | Unique pairs used | Supported images | Unavailable images | All-draw acceptance |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for name, r in summary['ordinal'].items():
        lines.append(f"| {name} | {percent(r['accuracy'])} | {r['valid_pairs']} | {r['unique_pairs']} | "
                     f"{r['supported_images']} | {r['unavailable_images']} | {percent(r['all_images_accepted_draw_fraction'])} |")
    lines.extend(['', 'T/T 描述透明区域内部的采样深度排序，T/B 描述局部透明区域与背景的排序。',
        'B/B 和全图结果提供参照；不存在由 mask 预设背景一定更远的规则。GT 决定正确顺序，预测不做 GT 缩放。',
        '像素对按有放回 Monte Carlo 抽样；去重数量和端点覆盖率在 CSV 中报告。大量重复抽样不等于独立统计证据。', ''])
    p = summary['penetration']
    if p['status'] == 'available':
        lines.extend([f"局部前表面诊断使用 {p['supported_images']} 张图像的 {p['valid_pairs']} 个采样对（{p['unique_pairs']} 个不同像素对）：",
            f"顺序反转率 **{percent(p['reversal_rate'])}**；严重间隔压缩率 **{percent(p['compression_rate'])}**；",
            f"两者并集为 **{percent(p['error_rate'])}**。"])
    else:
        lines.append('局部前表面诊断 unavailable：没有达到最低支持条件的逐图比较，不能据此评价穿透率。')
    lines.extend(['', '这里只保留 GT 证实透明表面在邻近背景前方的像素对；压缩比较使用 log 深度间隔，因此对整体正比例缩放不敏感。',
        '**邻近背景不是同一相机射线上真实被遮挡的背景；这些结果不能直接证明或排除物理 background penetration。**', '',
        '### Transparent Boundary Analysis', '',
        f"距离变换定义内部、边界带和背景环带。边界带宽 {protocol['boundary_width']} px，排除边界附近 "
        f"{protocol['boundary_margin']} px，背景半径 {protocol['ring_radius']} px。",
        '对图像边界、没有可靠内部或背景的细碎 mask、有效像素不足的情况，不输出有效边界对比率。', '',
        '| Status | Supported images | Valid pairs | Normalized contrast MAE | Log contrast MAE | Reversal rate |',
        '|---|---:|---:|---:|---:|---:|'])
    b = summary['boundary']
    lines.append(f"| {b['status']} | {b['supported_images']} | {b['valid_pairs']} | "
                 f"{number(b['normalized_contrast_mae'])} | {number(b['log_contrast_mae'])} | {percent(b['reversal_rate'])} |")
    lines.extend(['', '归一化对比度定义为 `2*(B-T)/(B+T)`，log 对比度为 `ln(B)-ln(T)`；均不依赖整体尺度。',
        '各区域的原始误差、背景对齐残差和 log 统计见 `boundary_analysis.csv`。空间阈值按原始像素定义，跨相机分辨率解释时需注意这一点。', '',
        '## D. Hardest Samples', '',
        '下表只对可用结果排序。ordinal/penetration 支持不足的图像不会被错误地列为最好或最坏。', '',
        '| Criterion | Sample ID | Value | Figure |', '|---|---|---:|---|'])
    for criterion, samples in summary['hardest_samples'].items():
        if not samples:
            lines.append(f'| {criterion} | unavailable | unavailable | — |')
        for item in samples:
            lines.append(f"| {criterion} | {item['sample_id']} | {number(item['value'])} | [view]({item['figure']}) |")
    lines.extend(['', '## Figures and provenance', '',
        '![Scale ratios](figures/scale_ratio_distribution.png)', '',
        '![Raw versus oracle](figures/raw_vs_aligned_absrel.png)', '',
        '![Background-aligned residuals](figures/transparent_vs_background.png)', '',
        '![Ordinal accuracy](figures/ordinal_accuracy.png)', '',
        '![Local order and compression](figures/penetration_rates.png)', '',
        '![Hardest samples](figures/hardest_samples.png)', '',
        '每张图的九面板均保存在可读样本路径下。Raw 预测完整显示，GT 缺测只遮罩 GT/误差；',
        '对齐深度和误差均标注 **GT-Dependent Diagnostic Only**。顺序/边界热图只在有足够支持的采样透明端点显示；灰色包含未采样位置。', '',
        f"Manifest SHA-256: `{summary['manifest_sha256']}`。",
        '逐样本预测、RGB、GT、mask 和相机来源校验见 `validation.json`；重算与历史基线差异见 `baseline_comparison.json`。',
        '该分析是项目自定义诊断协议，不应冒充官方 ClearGrasp benchmark。'])
    Path(path).write_text('\n'.join(lines)+'\n', encoding='utf-8')
