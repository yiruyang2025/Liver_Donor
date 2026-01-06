import json
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional
class AblationComparator:
    def __init__(self, ablation_dir: str = 'ablation_results', main_dir: str = 'results'):
        self.ablation_dir = Path(ablation_dir)
        self.main_dir = Path(main_dir)
        self.ablation_path = self.ablation_dir / 'ablation_results.json'
        self.main_path = self.main_dir / 'metrics.json'
    def load_results(self, path: Path) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)
    def validate_paths(self) -> bool:
        if not self.ablation_path.exists():
            print(f"Error: Ablation results not found at {self.ablation_path}")
            return False
        return True
    def compare_ablation_results(self) -> Tuple[Dict, str, float]:
        ablation_results = self.load_results(self.ablation_path)
        best_config = None
        best_accuracy = 0.0
        for config_name, metrics in ablation_results.items():
            acc = metrics['accuracy']
            if acc > best_accuracy:
                best_accuracy = acc
                best_config = config_name
        return ablation_results, best_config, best_accuracy
    def compare_with_main_pipeline(self, best_config: str, ablation_results: Dict) -> Optional[Dict]:
        if not self.main_path.exists():
            return None
        main_results = self.load_results(self.main_path)
        main_acc = main_results.get('accuracy', 0.0)
        best_ablation_acc = ablation_results[best_config]['accuracy']
        diff = main_acc - best_ablation_acc
        comparison = {
            'main_pipeline_accuracy': main_acc,
            'best_ablation_accuracy': best_ablation_acc,
            'absolute_difference': diff,
            'relative_difference_percent': (diff / best_ablation_acc * 100) if best_ablation_acc > 0 else 0.0,
            'winner': 'main_pipeline' if diff > 0.001 else ('ablation' if diff < -0.001 else 'tie'),
            'main_results': main_results
        }
        return comparison
    def format_results_table(self, ablation_results: Dict) -> pd.DataFrame:
        table_data = []
        for config_name, metrics in ablation_results.items():
            row = {
                'Model': config_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'AUC-ROC': f"{metrics.get('auc_roc', 0.0):.4f}" if isinstance(metrics.get('auc_roc'), float) else 'N/A',
                'Sensitivity': f"{metrics.get('sensitivity', 0.0):.4f}",
                'Specificity': f"{metrics.get('specificity', 0.0):.4f}",
                'Precision': f"{metrics.get('precision', 0.0):.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'TP': metrics['tp'],
                'TN': metrics['tn'],
                'FP': metrics['fp'],
                'FN': metrics['fn']
            }
            table_data.append(row)
        return pd.DataFrame(table_data)
    def print_results(self, ablation_results: Dict, best_config: str, best_accuracy: float):
        print("\n" + "=" * 130)
        print("TABLE 1: ABLATION STUDY RESULTS - LIVER TRANSPLANTABILITY PREDICTION")
        print("=" * 130)
        print(f"\n{'Model':<30} {'Accuracy':<12} {'AUC-ROC':<12} {'Sensitivity':<12} {'Specificity':<12} {'Precision':<12} {'F1':<12}")
        print("-" * 130)
        for config_name, metrics in ablation_results.items():
            acc = metrics['accuracy']
            auc = f"{metrics['auc_roc']:.4f}" if isinstance(metrics['auc_roc'], float) else 'N/A'
            sens = f"{metrics.get('sensitivity', 0.0):.4f}"
            spec = f"{metrics.get('specificity', 0.0):.4f}"
            prec = f"{metrics.get('precision', 0.0):.4f}"
            f1 = f"{metrics['f1']:.4f}"
            marker = " (BEST)" if config_name == best_config else ""
            print(f"{config_name:<30} {acc:<12.4f} {auc:<12} {sens:<12} {spec:<12} {prec:<12} {f1:<12}{marker}")
        print("-" * 130)
        print(f"\nBest Configuration: {best_config} with {best_accuracy:.4f} accuracy")
    def print_comparison(self, comparison: Dict):
        print("\n" + "=" * 130)
        print("TABLE 2: MAIN PIPELINE VS BEST ABLATION CONFIGURATION")
        print("=" * 130)
        main_acc = comparison['main_pipeline_accuracy']
        best_ablation_acc = comparison['best_ablation_accuracy']
        diff = comparison['absolute_difference']
        winner = comparison['winner']
        print(f"\nMain Pipeline Accuracy:          {main_acc:.4f}")
        print(f"Best Ablation Accuracy:          {best_ablation_acc:.4f}")
        print(f"Absolute Difference:             {diff:+.4f}")
        print(f"Relative Difference:             {comparison['relative_difference_percent']:+.2f}%")
        if winner == 'main_pipeline':
            print(f"\nResult: Main pipeline achieves {abs(diff):.2%} higher accuracy")
        elif winner == 'ablation':
            print(f"\nResult: Ablation baseline achieves {abs(diff):.2%} higher accuracy")
        else:
            print(f"\nResult: Both configurations achieve statistically equivalent performance")
    def export_to_csv(self, ablation_results: Dict):
        df = self.format_results_table(ablation_results)
        csv_path = self.ablation_dir / 'ablation_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults exported to {csv_path}")
        return df
    def generate_summary_statistics(self, ablation_results: Dict) -> Dict:
        accuracies = [metrics['accuracy'] for metrics in ablation_results.values()]
        auc_scores = [metrics.get('auc_roc', 0.0) for metrics in ablation_results.values() if isinstance(metrics.get('auc_roc'), float)]
        f1_scores = [metrics['f1'] for metrics in ablation_results.values()]
        summary = {
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'std_accuracy': (sum((x - sum(accuracies) / len(accuracies)) ** 2 for x in accuracies) / len(accuracies)) ** 0.5,
            'max_accuracy': max(accuracies),
            'min_accuracy': min(accuracies),
            'mean_auc': sum(auc_scores) / len(auc_scores) if auc_scores else 0.0,
            'mean_f1': sum(f1_scores) / len(f1_scores)
        }
        return summary
    def print_summary_statistics(self, summary: Dict):
        print("\n" + "=" * 130)
        print("SUMMARY STATISTICS ACROSS ALL CONFIGURATIONS")
        print("=" * 130)
        print(f"\nAccuracy - Mean: {summary['mean_accuracy']:.4f}, Std: {summary['std_accuracy']:.4f}")
        print(f"Accuracy - Max: {summary['max_accuracy']:.4f}, Min: {summary['min_accuracy']:.4f}")
        print(f"AUC-ROC - Mean: {summary['mean_auc']:.4f}")
        print(f"F1-Score - Mean: {summary['mean_f1']:.4f}")
    def run(self):
        if not self.validate_paths():
            return
        ablation_results, best_config, best_accuracy = self.compare_ablation_results()
        self.print_results(ablation_results, best_config, best_accuracy)
        comparison = self.compare_with_main_pipeline(best_config, ablation_results)
        if comparison:
            self.print_comparison(comparison)
        summary = self.generate_summary_statistics(ablation_results)
        self.print_summary_statistics(summary)
        self.export_to_csv(ablation_results)
        print("\n" + "=" * 130)
def main():
    parser = argparse.ArgumentParser(
        description='Ablation study comparison for liver transplantability prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example usage: python compare_results.py --ablation_dir ablation_results --main_dir results'
    )
    parser.add_argument('--ablation_dir', type=str, default='ablation_results', help='Directory containing ablation study results')
    parser.add_argument('--main_dir', type=str, default='results', help='Directory containing main pipeline results')
    args = parser.parse_args()
    comparator = AblationComparator(ablation_dir=args.ablation_dir, main_dir=args.main_dir)
    comparator.run()
if __name__ == '__main__':
    main()
