import typer
from pathlib import Path
from .pipeline import DatasetPipeline
from .export import DatasetExporter

app = typer.Typer(help="DatasetForge: create fine-tuning datasets from Kaggle, Hugging Face, GitHub, PDFs, and more.")


@app.command()
def ingest(
    source: str = typer.Option(..., help="Source connector name: kaggle, huggingface, github, pdf."),
    identifier: str = typer.Option(..., help="Identifier for the source, e.g. dataset name, repo URL, or file path."),
    output: Path = typer.Option(Path("outputs"), help="Output directory for generated dataset files."),
    format: str = typer.Option("jsonl", help="Export format: jsonl, alpaca, chatml."),
    split: bool = typer.Option(True, help="Split into train/validation/test sets."),
):
    """Ingest a dataset source and build a ready-to-train dataset."""
    output.mkdir(parents=True, exist_ok=True)
    pipeline = DatasetPipeline(source=source, identifier=identifier)
    dataset = pipeline.run()
    exporter = DatasetExporter(dataset)

    if split:
        exporter.split_and_export(output, fmt=format)
    else:
        exporter.export(output / f"{identifier.replace('/', '_')}.{format}", fmt=format)

    typer.echo(f"DatasetForge ingestion completed: {output}")


@app.command()
def validate(
    input_path: Path = typer.Argument(..., help="Path to a dataset file to validate."),
):
    """Validate dataset schema, deduplication, and output format."""
    exporter = DatasetExporter([])
    exporter.validate(input_path)
    typer.echo(f"Validated dataset: {input_path}")


@app.command()
def version():
    """Show DatasetForge version."""
    from . import __version__

    typer.echo(__version__)
