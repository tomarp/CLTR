from __future__ import annotations

from pathlib import Path
import json
import shutil

from .config import CLTRConfig, default_config
from .analysis import CLTRAnalyzer
from .dataset import CLTRDataset
from .preprocessing import SessionPreprocessor
from .reporting import ReportWriter
from .review import evaluate_report_quality
from .utils import ensure_dir, set_mplconfigdir, write_json


class CLTRPipeline:
    def __init__(self, dataset_root: str | Path, outdir: str | Path, config: CLTRConfig | None = None):
        self.dataset_root = Path(dataset_root)
        self.outdir = Path(outdir)
        self.config = config or default_config()
        ensure_dir(self.outdir)
        set_mplconfigdir(self.outdir, self.config.runtime.mpl_config_dirname)
        self.dataset = CLTRDataset(self.dataset_root, self.config)
        self.analyzer = CLTRAnalyzer(self.config)

    def _prepare_run_dirs(self) -> None:
        managed = [
            self.outdir / self.config.output.validation_dir,
            self.outdir / self.config.output.manifests_dir,
            self.outdir / self.config.output.data_dir,
            self.outdir / self.config.output.analysis_dir,
            self.outdir / self.config.output.report_dir,
            self.outdir / "cache",
        ]
        for path in managed:
            if path.exists():
                shutil.rmtree(path)
        summary = self.outdir / "run_summary.json"
        if summary.exists():
            summary.unlink()

    def validate(self) -> dict:
        outdir = ensure_dir(self.outdir / self.config.output.validation_dir)
        return self.dataset.write_validation(outdir)

    def _resolve_session_selection(
        self,
        *,
        session_limit: int | None = None,
        session_ids: list[str] | None = None,
        participants: list[str] | None = None,
        conditions: list[str] | None = None,
    ) -> list[str]:
        bundle = self.dataset.load_bundle()
        manifest = bundle.manifest.copy()
        if not bundle.sessions.empty and "session_id" in bundle.sessions.columns:
            allowed = set(bundle.sessions["session_id"].astype(str))
            manifest = manifest.loc[manifest["session_id"].astype(str).isin(allowed)].copy()
        else:
            manifest = manifest.loc[manifest["in_timeline"]].copy()
        if session_ids:
            wanted = {str(x) for x in session_ids}
            manifest = manifest.loc[manifest["session_id"].isin(wanted)].copy()
        if participants:
            wanted = {str(x) for x in participants}
            manifest = manifest.loc[manifest["participant_id"].isin(wanted)].copy()
        if conditions:
            wanted = {str(x) for x in conditions}
            manifest = manifest.loc[manifest["condition_code"].isin(wanted)].copy()
        if session_limit is not None:
            manifest = manifest.head(session_limit).copy()
        return manifest["session_id"].astype(str).tolist()

    def preprocess(
        self,
        session_limit: int | None = None,
        session_ids: list[str] | None = None,
        participants: list[str] | None = None,
        conditions: list[str] | None = None,
    ) -> dict:
        bundle = self.dataset.load_bundle()
        selected_session_ids = self._resolve_session_selection(
            session_limit=session_limit,
            session_ids=session_ids,
            participants=participants,
            conditions=conditions,
        )
        manifest_out = ensure_dir(self.outdir / self.config.output.manifests_dir)
        data_root = ensure_dir(self.outdir / self.config.output.data_dir / self.config.output.session_dir)
        manifest = bundle.manifest.loc[bundle.manifest["session_id"].isin(selected_session_ids)].copy()
        manifest.to_csv(manifest_out / "session_manifest.csv", index=False)
        preprocessor = SessionPreprocessor(self.dataset_root, self.config, bundle)
        session_artifacts = []
        for session_id in selected_session_ids:
            artifacts = preprocessor.process_session(session_id)
            session_root = ensure_dir(data_root / session_id)
            artifacts.aligned_minute.to_csv(session_root / "minute_aligned_features.csv", index=False)
            artifacts.phase_summary.to_csv(session_root / "phase_summary.csv", index=False)
            (session_root / "processing_metadata.json").write_text(
                json.dumps(artifacts.processing_metadata, indent=2),
                encoding="utf-8",
            )
            session_artifacts.append({
                "session_id": session_id,
                "aligned_df": artifacts.aligned_minute,
                "phase_df": artifacts.phase_summary,
                "processing_metadata": artifacts.processing_metadata,
            })
        return {"manifest": manifest, "session_inputs": session_artifacts}

    def analyze(
        self,
        preprocessed: dict | None = None,
        session_limit: int | None = None,
        session_ids: list[str] | None = None,
        participants: list[str] | None = None,
        conditions: list[str] | None = None,
    ) -> dict:
        if preprocessed is None:
            preprocessed = self.preprocess(
                session_limit=session_limit,
                session_ids=session_ids,
                participants=participants,
                conditions=conditions,
            )
        session_minutes = [s["aligned_df"] for s in preprocessed["session_inputs"]]
        phase_summaries = [s["phase_df"] for s in preprocessed["session_inputs"]]
        cohort_outputs = self.analyzer.build_cohort_outputs(session_minutes, phase_summaries)
        analysis_root = ensure_dir(self.outdir / self.config.output.analysis_dir / self.config.output.cohort_dir)
        for name, value in cohort_outputs.items():
            value.to_csv(analysis_root / f"{name}.csv", index=False)
        return cohort_outputs | {"preprocessed": preprocessed}

    def report(
        self,
        analyzed: dict | None = None,
        session_limit: int | None = None,
        session_ids: list[str] | None = None,
        participants: list[str] | None = None,
        conditions: list[str] | None = None,
        modalities: list[str] | None = None,
    ) -> dict:
        if analyzed is None:
            analyzed = self.analyze(
                session_limit=session_limit,
                session_ids=session_ids,
                participants=participants,
                conditions=conditions,
            )
        report_root = self.outdir / self.config.output.report_dir
        if report_root.exists():
            shutil.rmtree(report_root)
        writer = ReportWriter(self.outdir, self.dataset_root, self.config)
        session_reports = []
        for session_inputs in analyzed["preprocessed"]["session_inputs"]:
            session_reports.append(writer.write_session_report(session_inputs, modalities=modalities))
        cohort_payload = {k: v for k, v in analyzed.items() if k != "preprocessed"}
        cohort_report = writer.write_cohort_report(cohort_payload, modalities=modalities)
        index_report = writer.write_all_sessions_index(analyzed["preprocessed"]["manifest"], session_reports, cohort_report)
        review_report = evaluate_report_quality(
            self.outdir,
            analyzed["preprocessed"]["manifest"],
            session_reports,
            cohort_report,
            analyzed=cohort_payload,
            config=self.config,
        )
        return {"session_reports": session_reports, "cohort_report": cohort_report, "index_report": index_report, "review_report": review_report}

    def run_all(
        self,
        session_limit: int | None = None,
        session_ids: list[str] | None = None,
        participants: list[str] | None = None,
        conditions: list[str] | None = None,
        modalities: list[str] | None = None,
    ) -> dict:
        self._prepare_run_dirs()
        validation = self.validate()
        preprocessed = self.preprocess(
            session_limit=session_limit,
            session_ids=session_ids,
            participants=participants,
            conditions=conditions,
        )
        analyzed = self.analyze(preprocessed=preprocessed)
        reports = self.report(analyzed=analyzed, modalities=modalities)
        summary = {
            "validation": validation,
            "n_sessions": len(preprocessed["session_inputs"]),
            "cohort_report": reports["cohort_report"],
            "index_report": reports["index_report"],
            "review_report": reports["review_report"],
        }
        write_json(summary, self.outdir / "run_summary.json")
        return summary
