"""
Evaluation utilities for Red Light Violation Detection.

Expose `ModelEvaluator` for programmatic use; CLI wrappers should import this.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

from red_light.config import (
    load_data_config,
    load_evaluation_config,
    validate_data_yaml_path,
    validate_model_path,
)


class ModelEvaluator:
    """Production-ready evaluator for red light detection models."""

    def __init__(self, model_path, data_yaml, eval_config_path):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model weights
            data_yaml: Path to data configuration file
            eval_config_path: Path to evaluation configuration file
        """
        self.model_path = validate_model_path(model_path)
        self.data_yaml = validate_data_yaml_path(data_yaml)
        self.eval_config = load_evaluation_config(eval_config_path)
        self.data_base_dir = self.data_yaml.parent

        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))

        self.data_config = load_data_config(self.data_yaml)
        self.class_names = self.data_config['names']

        self.output_dir = self.model_path.parent.parent / 'evaluation'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Evaluation results will be saved to: {self.output_dir}")

    def evaluate(self, split='val'):
        """
        Run evaluation on specified split.

        Args:
            split: Dataset split to evaluate ('val' or 'test')

        Returns:
            Dictionary containing evaluation metrics
        """
        eval_params = self.eval_config['evaluation']

        print("\n" + "=" * 60)
        print(f"Evaluating on {split} set")
        print("=" * 60)

        results = self.model.val(
            data=str(self.data_yaml),
            split=split,
            save_json=eval_params['save_json'],
            save_hybrid=eval_params['save_hybrid'],
            conf=eval_params['conf_threshold'],
            iou=eval_params['iou_threshold'],
            max_det=eval_params['max_detections'],
            plots=True,
            verbose=eval_params['verbose']
        )

        metrics = self._extract_metrics(results)

        self._print_metrics_summary(metrics)
        roi_metrics = self._compute_roi_metrics(split, eval_params)
        if roi_metrics:
            self._print_roi_summary(roi_metrics)
        self._generate_class_report(metrics)
        self._save_metrics(metrics, roi_metrics, split)
        self._generate_visualizations(metrics, split)
        self._log_tensorboard(metrics, roi_metrics, split)

        return metrics

    def _extract_metrics(self, results):
        """Extract metrics from YOLO results."""
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'per_class': {}
        }

        for idx, class_name in enumerate(self.class_names):
            if idx < len(results.box.ap_class_index):
                class_idx = np.where(results.box.ap_class_index == idx)[0]

                if len(class_idx) > 0:
                    metrics['per_class'][class_name] = {
                        'ap50': float(results.box.ap50[class_idx[0]]) if len(results.box.ap50) > class_idx[0] else 0.0,
                        'ap50-95': float(results.box.ap[class_idx[0]]) if len(results.box.ap) > class_idx[0] else 0.0,
                    }
                else:
                    metrics['per_class'][class_name] = {
                        'ap50': 0.0,
                        'ap50-95': 0.0,
                    }
            else:
                metrics['per_class'][class_name] = {
                    'ap50': 0.0,
                    'ap50-95': 0.0,
                }

        return metrics

    def _print_metrics_summary(self, metrics):
        """Print formatted metrics summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"mAP@0.5:      {metrics['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print("=" * 60)

    def _print_roi_summary(self, roi_metrics):
        """Print ROI metrics summary if available."""
        print("\n" + "=" * 60)
        print("ROI EVALUATION (Stop-line band)")
        print("=" * 60)
        print(f"Precision:    {roi_metrics['precision']:.4f}")
        print(f"Recall:       {roi_metrics['recall']:.4f}")
        print(f"F1:           {roi_metrics['f1']:.4f}")
        print("=" * 60)
        print("Per-class (ROI)")
        print(f"{'Class':<15} {'P':>8} {'R':>8} {'F1':>8}")
        print("-" * 60)
        for cls, m in roi_metrics['per_class'].items():
            print(
                f"{cls:<15} {m['precision']:.4f} {m['recall']:.4f} {m['f1']:.4f}")
        print("=" * 60)

    def _generate_class_report(self, metrics):
        """Generate detailed per-class performance report."""
        viz_config = self.eval_config['visualization']
        threshold = viz_config['low_performance_threshold']

        print("\n" + "=" * 60)
        print("PER-CLASS PERFORMANCE")
        print("=" * 60)
        print(f"{'Class':<15} {'AP@0.5':>10} {'AP@0.5:0.95':>14}")
        print("-" * 60)

        sorted_classes = sorted(
            metrics['per_class'].items(),
            key=lambda x: x[1]['ap50-95'],
            reverse=True
        )

        for class_name, class_metrics in sorted_classes:
            print(
                f"{class_name:<15} {class_metrics['ap50']:>10.4f} {class_metrics['ap50-95']:>14.4f}")

        print("=" * 60)

        print("\n" + "=" * 60)
        print(f"CLASSES NEEDING ATTENTION (AP@0.5:0.95 < {threshold})")
        print("=" * 60)

        low_performing = [
            (name, metrics_dict['ap50-95'])
            for name, metrics_dict in metrics['per_class'].items()
            if metrics_dict['ap50-95'] < threshold
        ]

        if low_performing:
            for name, ap in sorted(low_performing, key=lambda x: x[1]):
                print(f"  - {name}: {ap:.4f}")
        else:
            print(f"  None - All classes performing above {threshold}!")
        print("=" * 60)

    def _save_metrics(self, metrics, roi_metrics, split):
        """Save metrics to JSON file."""
        output_path = self.output_dir / f'metrics_{split}.json'

        metrics_to_save = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'split': split,
            'config': self.eval_config,
            'metrics': metrics,
            'roi_metrics': roi_metrics
        }

        with open(output_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)

        print(f"\nMetrics saved to: {output_path}")

    def _generate_visualizations(self, metrics, split):
        """Generate visualization plots."""
        print("\nGenerating visualizations...")

        self._plot_per_class_ap(metrics, split)
        self._plot_performance_heatmap(metrics, split)

        print("Visualizations saved!")

    def _plot_per_class_ap(self, metrics, split):
        """Plot per-class Average Precision comparison."""
        viz_config = self.eval_config['visualization']
        bar_config = viz_config['bar_chart']

        fig, ax = plt.subplots(figsize=tuple(bar_config['figsize']))

        classes = list(metrics['per_class'].keys())
        ap50 = [metrics['per_class'][c]['ap50'] for c in classes]
        ap50_95 = [metrics['per_class'][c]['ap50-95'] for c in classes]

        x = np.arange(len(classes))
        width = bar_config['bar_width']

        ax.bar(
            x - width / 2, ap50, width,
            label='AP@0.5',
            alpha=0.8,
            color=bar_config['colors']['ap50']
        )
        ax.bar(
            x + width / 2, ap50_95, width,
            label='AP@0.5:0.95',
            alpha=0.8,
            color=bar_config['colors']['ap50_95']
        )

        ax.set_xlabel('Class', fontweight='bold')
        ax.set_ylabel('Average Precision', fontweight='bold')
        ax.set_title(
            f'Per-Class Average Precision - {split.upper()} Set', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=bar_config['grid_alpha'])
        ax.set_ylim(bar_config['y_limit'])

        plt.tight_layout()
        output_path = self.output_dir / \
            f'per_class_ap_{split}.{viz_config["format"]}'
        plt.savefig(output_path, dpi=viz_config['dpi'], bbox_inches='tight')
        plt.close()

    def _plot_performance_heatmap(self, metrics, split):
        """Plot performance heatmap."""
        viz_config = self.eval_config['visualization']
        heatmap_config = viz_config['heatmap']

        fig, ax = plt.subplots(figsize=tuple(heatmap_config['figsize']))

        classes = list(metrics['per_class'].keys())
        data = np.array([
            [metrics['per_class'][c]['ap50'] for c in classes],
            [metrics['per_class'][c]['ap50-95'] for c in classes]
        ])

        sns.heatmap(
            data,
            annot=True,
            fmt=heatmap_config['annot_format'],
            cmap=viz_config['heatmap_cmap'],
            xticklabels=classes,
            yticklabels=['AP@0.5', 'AP@0.5:0.95'],
            vmin=heatmap_config['vmin'],
            vmax=heatmap_config['vmax'],
            ax=ax,
            cbar_kws={'label': 'Average Precision'}
        )

        ax.set_title(
            f'Performance Heatmap - {split.upper()} Set', fontweight='bold', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_path = self.output_dir / \
            f'performance_heatmap_{split}.{viz_config["format"]}'
        plt.savefig(output_path, dpi=viz_config['dpi'], bbox_inches='tight')
        plt.close()

    def _log_tensorboard(self, metrics, roi_metrics, split):
        """Log global and ROI metrics to TensorBoard."""
        log_dir = self.output_dir / 'tensorboard' / split
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))

        # Global metrics
        writer.add_scalar('global/mAP50', metrics['mAP50'], 0)
        writer.add_scalar('global/mAP50_95', metrics['mAP50-95'], 0)
        writer.add_scalar('global/precision', metrics['precision'], 0)
        writer.add_scalar('global/recall', metrics['recall'], 0)
        for cls, m in metrics['per_class'].items():
            writer.add_scalar(f'global/per_class/{cls}/ap50', m['ap50'], 0)
            writer.add_scalar(f'global/per_class/{cls}/ap50_95',
                              m['ap50-95'], 0)

        # ROI metrics (optional)
        if roi_metrics:
            writer.add_scalar('roi/precision', roi_metrics['precision'], 0)
            writer.add_scalar('roi/recall', roi_metrics['recall'], 0)
            writer.add_scalar('roi/f1', roi_metrics['f1'], 0)
            writer.add_scalar('roi/band/y_min', roi_metrics['roi']['y_min'], 0)
            writer.add_scalar('roi/band/y_max', roi_metrics['roi']['y_max'], 0)
            for cls, m in roi_metrics['per_class'].items():
                writer.add_scalar(
                    f'roi/per_class/{cls}/precision', m['precision'], 0)
                writer.add_scalar(
                    f'roi/per_class/{cls}/recall', m['recall'], 0)
                writer.add_scalar(
                    f'roi/per_class/{cls}/f1', m['f1'], 0)

        writer.flush()
        writer.close()

    # -------------------------------------------------------------------------
    # ROI evaluation helpers
    # -------------------------------------------------------------------------
    def _compute_roi_metrics(self, split, eval_params):
        """Compute ROI precision/recall over a stop-line band."""
        roi_cfg = self.eval_config.get('roi', {})
        if not roi_cfg or not roi_cfg.get('enabled', False):
            return None

        image_dir, label_dir = self._resolve_split_dirs(split)
        if not image_dir.exists():
            print(f"ROI evaluation skipped: image dir not found {image_dir}")
            return None
        if not label_dir.exists():
            print(f"ROI evaluation skipped: label dir not found {label_dir}")
            return None

        preds_by_image = self._run_predictions_for_split(
            image_dir,
            eval_params,
        )
        gt_by_image = self._load_labels_for_split(label_dir)

        class_ids = [
            self.class_names.index(c)
            for c in roi_cfg.get('classes', [])
            if c in self.class_names
        ]
        if not class_ids:
            print("ROI evaluation skipped: no valid classes configured.")
            return None

        iou_thr = float(roi_cfg.get('iou_threshold', 0.5))
        y_min = float(roi_cfg.get('y_min', 0.0))
        y_max = float(roi_cfg.get('y_max', 1.0))

        per_class = {}
        total_tp = total_fp = total_fn = 0

        for class_id in class_ids:
            cls_name = self.class_names[class_id]

            cls_gt = []
            cls_pred = []

            for stem, labels in gt_by_image.items():
                for label in labels:
                    if label['class_id'] != class_id:
                        continue
                    if self._in_roi(label['box'], y_min, y_max):
                        cls_gt.append(label['box'])

                preds = preds_by_image.get(stem, [])
                for pred in preds:
                    if pred['class_id'] != class_id:
                        continue
                    if self._in_roi(pred['box'], y_min, y_max):
                        cls_pred.append(pred)

            tp, fp, fn = self._match_detections(cls_gt, cls_pred, iou_thr)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision, recall, f1 = self._prf(tp, fp, fn)
            per_class[cls_name] = {
                'precision': precision, 'recall': recall, 'f1': f1}

        precision, recall, f1 = self._prf(total_tp, total_fp, total_fn)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class': per_class,
            'roi': {'y_min': y_min, 'y_max': y_max}
        }

    def _resolve_split_dirs(self, split):
        """Resolve image/label directories for a split."""
        key = 'val' if split == 'val' else 'test'
        image_dir = Path(self.data_config[key])
        if not image_dir.is_absolute():
            image_dir = self.data_base_dir / image_dir
        label_dir = image_dir.parent / 'labels'
        return image_dir, label_dir

    def _run_predictions_for_split(self, image_dir, eval_params):
        """Run model prediction over a directory and return per-image boxes."""
        predictions = {}
        for res in self.model.predict(
            source=str(image_dir),
            stream=True,
            conf=eval_params['conf_threshold'],
            iou=eval_params['iou_threshold'],
            max_det=eval_params['max_detections'],
            verbose=False,
        ):
            stem = Path(res.path).stem
            h, w = res.orig_shape
            if h == 0 or w == 0:
                continue
            preds = []
            xyxy = res.boxes.xyxy.cpu()
            cls = res.boxes.cls.cpu()
            conf = res.boxes.conf.cpu()
            for idx in range(len(cls)):
                x1, y1, x2, y2 = xyxy[idx].tolist()
                preds.append({
                    'class_id': int(cls[idx].item()),
                    'conf': float(conf[idx].item()),
                    'box': [
                        max(0.0, x1 / w),
                        max(0.0, y1 / h),
                        min(1.0, x2 / w),
                        min(1.0, y2 / h),
                    ]
                })
            predictions[stem] = preds
        return predictions

    def _load_labels_for_split(self, label_dir):
        """Load YOLO-format labels for a split."""
        labels = {}
        for lbl_path in Path(label_dir).glob('*.txt'):
            stem = lbl_path.stem
            boxes = []
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = max(0.0, xc - w / 2)
                    y1 = max(0.0, yc - h / 2)
                    x2 = min(1.0, xc + w / 2)
                    y2 = min(1.0, yc + h / 2)
                    boxes.append({'class_id': cls_id, 'box': [x1, y1, x2, y2]})
            labels[stem] = boxes
        return labels

    @staticmethod
    def _in_roi(box, y_min, y_max):
        """Check if box center falls inside the ROI band."""
        _, y1, _, y2 = box
        y_center = (y1 + y2) / 2
        return y_min <= y_center <= y_max

    @staticmethod
    def _iou(box_a, box_b):
        """Compute IoU for two boxes in normalized xyxy."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _match_detections(self, gt_boxes, pred_boxes, iou_thr):
        """Greedy match predictions to GT boxes for a class."""
        matched_gt = set()
        tp = fp = 0
        for pred in sorted(pred_boxes, key=lambda x: x['conf'], reverse=True):
            best_iou = 0.0
            best_idx = None
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = self._iou(pred['box'], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_thr and best_idx is not None:
                matched_gt.add(best_idx)
                tp += 1
            else:
                fp += 1
        fn = len(gt_boxes) - len(matched_gt)
        return tp, fp, fn

    @staticmethod
    def _prf(tp, fp, fn):
        """Compute precision, recall, f1."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
