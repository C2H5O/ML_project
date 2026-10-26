"""Raw TransCG metric statistics; GT validity and object regions stay separate."""
import numpy as np
from src.evaluation.metrics import MetricAccumulator


class TransCGMetrics:
    def __init__(self):
        self.regions = {name: MetricAccumulator() for name in
                        ("full_image", "transparent_objects", "non_transparent_regions")}
        self.per_image = []
        self.skipped = []

    def update(self, sample_id, prediction, gt, mask, annotations):
        if gt is None:
            self.skipped.append(dict(sample_id=sample_id, reason="Refined GT unavailable", annotations=annotations))
            return None
        regions = {"full_image": np.ones(gt.shape, bool)}
        if mask is not None:
            regions.update(transparent_objects=mask, non_transparent_regions=~mask)
        else:
            self.skipped.append(dict(sample_id=sample_id, reason="No reliable transparent-object mask; full image only",
                                     annotations=annotations))
        row = dict(sample_id=sample_id, regions={}, annotations=annotations)
        for name, region in regions.items():
            score = MetricAccumulator().update(prediction, gt, region).result()
            self.regions[name].update(prediction, gt, region)
            count = int((region & np.isfinite(gt) & (gt > 0)).sum())
            row["regions"][name] = dict(**score, valid_gt_pixels=count,
                prediction_coverage=score["valid_pixels"] / count if count else None)
        self.per_image.append(row)
        return row

    def result(self):
        return dict(protocol="raw metric, positive finite GT/prediction intersection; no GT alignment",
            gt_alignment=False, scale_source="calibrated RGB intrinsics only; native focal / 300",
            aggregation="global pixel-weighted, RMSE from pooled squared errors", units="metres",
            region_definition="mask nonzero = transparent objects; zero = non-transparent/background; GT validity is independent",
            regions={name: value.result() for name, value in self.regions.items()},
            per_image=self.per_image, skipped=self.skipped)
