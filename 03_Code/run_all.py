"""Rebuild from cached data. This does not redownload prices or place orders."""
import analyze_optiver
import analyze_strategy
import verify_project
import build_reports
import build_notebook

if __name__=='__main__':
    analyze_optiver.run()
    analyze_strategy.run()
    verify_project.run()
    build_reports.main()
    build_notebook.main()
    print('Research, checks, reports and notebook rebuilt from cached snapshots.')
