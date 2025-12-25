import os
from pathlib import Path

from src.reporting import generar_informe_tecnico_pdf


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / 'output' / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = str(output_dir / 'full_report.pdf')

    # Gather appendix files (common source files)
    src_dir = repo_root / 'src'
    candidates = ['main.py', 'fuzzy_module.py', 'ml_module.py', 'reporting.py', 'web_app.py', 'utils.py']
    files_for_appendix = {}
    for c in candidates:
        p = src_dir / c
        if p.exists():
            files_for_appendix[c] = str(p)

    # Dataset path: prefer data/german_credit_data.csv then root
    dataset_path = None
    d1 = repo_root / 'data' / 'german_credit_data.csv'
    d2 = repo_root / 'german_credit_data.csv'
    if d1.exists():
        dataset_path = str(d1)
    elif d2.exists():
        dataset_path = str(d2)

    # Collect up to 3 figure paths from output/figures
    figure_dir = repo_root / 'output' / 'figures'
    figure_paths = []
    if figure_dir.exists():
        for i, fp in enumerate(sorted(figure_dir.glob('*'))):
            if i >= 6:
                break
            figure_paths.append(str(fp))

    ok = generar_informe_tecnico_pdf(pdf_path, files_for_appendix=files_for_appendix, dataset_path=dataset_path, figure_paths=figure_paths)
    if ok:
        print(f'Report generated: {pdf_path}')
        return 0
    else:
        print('Failed to generate report')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
