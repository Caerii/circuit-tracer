import argparse
import logging
import os
import time
import warnings

# CLI defaults
DEFAULT_MAX_N_LOGITS = 10
DEFAULT_DESIRED_LOGIT_PROB = 0.95
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_FEATURE_NODES = 7500
DEFAULT_NODE_THRESHOLD = 0.8
DEFAULT_EDGE_THRESHOLD = 0.98
DEFAULT_SERVER_PORT = 8041


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="CLI for attribution, graph file creation, and server hosting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Attribution subcommand
    attr_parser = subparsers.add_parser("attribute", help="Run attribution analysis on a prompt")

    # Arguments from attribute_batch.py
    attr_parser.add_argument(
        "-m",
        "--model",
        type=str,
        help=("Model architecture to use for attribution. Can be inferred from transcoder config."),
    )
    attr_parser.add_argument(
        "-t",
        "--transcoder_set",
        required=True,
        help=(
            "HuggingFace repository ID containing transcoders "
            "(e.g. username/repo-name, username/repo-name@revision)."
        ),
    )
    attr_parser.add_argument("-p", "--prompt", required=True, help="Input prompt text to analyze.")
    attr_parser.add_argument(
        "-o",
        "--graph_output_path",
        help=(
            "Path where to save the attribution graph (.pt file). Required if not "
            "creating graph files."
        ),
    )
    attr_parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
        default="float32",
        help="Data type for model weights (default: float32).",
    )
    attr_parser.add_argument(
        "--max_n_logits",
        type=int,
        default=DEFAULT_MAX_N_LOGITS,
        help="Maximum number of logit nodes.",
    )
    attr_parser.add_argument(
        "--desired_logit_prob",
        type=float,
        default=DEFAULT_DESIRED_LOGIT_PROB,
        help="Cumulative probability threshold for top logits.",
    )
    attr_parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for backward passes."
    )
    attr_parser.add_argument(
        "--offload",
        choices=["cpu", "disk", None],
        default=None,
        help="Offload model parameters to save memory.",
    )
    attr_parser.add_argument(
        "--max_feature_nodes",
        type=int,
        default=DEFAULT_MAX_FEATURE_NODES,
        help="Maximum number of feature nodes.",
    )
    attr_parser.add_argument("--verbose", action="store_true", help="Display progress information.")
    attr_parser.add_argument(
        "--lazy-encoder",
        action="store_true",
        help="Enable lazy loading for encoder weights to save memory.",
    )
    attr_parser.add_argument(
        "--lazy-decoder",
        action="store_true",
        default=True,
        help="Enable lazy loading for decoder weights to save memory (default: True).",
    )
    attr_parser.add_argument(
        "--backend",
        type=str,
        choices=["transformerlens", "nnsight"],
        default="transformerlens",
        help="Backend to use for the replacement model (default: transformerlens).",
    )

    # Arguments for graph creation
    attr_parser.add_argument(
        "--slug",
        type=str,
        help=(
            "Slug for the model metadata (used for graph files). Required if creating "
            "graph files or starting server."
        ),
    )
    attr_parser.add_argument(
        "--graph_file_dir",
        type=str,
        help=(
            "Path to save the output JSON graph files, and also used as data dir for "
            "server. Required if creating graph files or starting server."
        ),
    )
    attr_parser.add_argument(
        "--node_threshold",
        type=float,
        default=DEFAULT_NODE_THRESHOLD,
        help="Node threshold for pruning graph files.",
    )
    attr_parser.add_argument(
        "--edge_threshold",
        type=float,
        default=DEFAULT_EDGE_THRESHOLD,
        help="Edge threshold for pruning graph files.",
    )

    # Server arguments
    attr_parser.add_argument(
        "--server",
        action="store_true",
        help="Start a local server to visualize graphs after processing.",
    )
    attr_parser.add_argument(
        "--port", type=int, default=DEFAULT_SERVER_PORT, help="Port for the local server."
    )
    attr_parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="Path to the directory containing feature files for local server, if using local transcoders (default: None)",
    )

    # Start-server subcommand
    server_parser = subparsers.add_parser(
        "start-server", help="Start a local server to visualize existing graphs"
    )
    server_parser.add_argument(
        "--graph_file_dir",
        type=str,
        required=True,
        help="Path to the directory containing graph JSON files.",
    )
    server_parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="Path to the directory containing feature files for local server, if using local transcoders (default: None)",
    )
    server_parser.add_argument(
        "--port", type=int, default=DEFAULT_SERVER_PORT, help="Port for the local server."
    )

    # Summarize subcommand — analysis API parity
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Emit a circuit-tracer.summary.v1 JSON document from a saved .pt graph",
    )
    summarize_parser.add_argument(
        "-g",
        "--graph",
        required=True,
        help="Path to a Graph .pt file produced by attribution.",
    )
    summarize_parser.add_argument(
        "-o",
        "--output",
        help="Optional output JSON path (default: stdout).",
    )
    summarize_parser.add_argument("--top-n", type=int, default=10, help="Number of top features.")
    summarize_parser.add_argument(
        "--node-threshold",
        type=float,
        default=DEFAULT_NODE_THRESHOLD,
        help="Pruning node threshold (set with --no-pruning to omit).",
    )
    summarize_parser.add_argument(
        "--edge-threshold",
        type=float,
        default=DEFAULT_EDGE_THRESHOLD,
        help="Pruning edge threshold (set with --no-pruning to omit).",
    )
    summarize_parser.add_argument(
        "--no-pruning",
        action="store_true",
        help="Omit pruning statistics from the summary.",
    )

    interventions_parser = subparsers.add_parser(
        "interventions",
        help="Emit a circuit-tracer.interventions.v1 plan JSON from a saved .pt graph",
    )
    interventions_parser.add_argument(
        "-g",
        "--graph",
        required=True,
        help="Path to a Graph .pt file produced by attribution.",
    )
    interventions_parser.add_argument(
        "-o",
        "--output",
        help="Optional output JSON path (default: stdout).",
    )
    interventions_parser.add_argument(
        "-n", type=int, default=10, help="Number of top features to intervene on."
    )
    interventions_parser.add_argument(
        "--value",
        type=float,
        default=0.0,
        help="Intervention value (0.0 = ablation).",
    )

    export_parser = subparsers.add_parser(
        "export-viz",
        help="Export a .pt graph to frontend JSON and print serve instructions",
    )
    export_parser.add_argument(
        "-g",
        "--graph",
        required=True,
        help="Path to a Graph .pt file produced by attribution.",
    )
    export_parser.add_argument("--slug", required=True, help="Slug / filename stem for the JSON.")
    export_parser.add_argument(
        "--graph_file_dir",
        required=True,
        help="Directory to write frontend JSON into.",
    )
    export_parser.add_argument(
        "--node_threshold", type=float, default=DEFAULT_NODE_THRESHOLD, help="Node prune threshold."
    )
    export_parser.add_argument(
        "--edge_threshold", type=float, default=DEFAULT_EDGE_THRESHOLD, help="Edge prune threshold."
    )
    export_parser.add_argument(
        "--neuronpedia-model",
        type=str,
        default=None,
        help="Optional Neuronpedia model id for export hints.",
    )
    export_parser.add_argument(
        "--neuronpedia-set",
        type=str,
        default=None,
        help="Optional Neuronpedia feature-set id for export hints.",
    )
    export_parser.add_argument(
        "-o",
        "--output",
        help="Optional path to write the viz-export metadata JSON (default: stdout).",
    )

    upload_parser = subparsers.add_parser(
        "upload-neuronpedia",
        help="Upload a graph (.pt or frontend .json) to Neuronpedia (requires API key)",
    )
    upload_parser.add_argument(
        "-g",
        "--graph",
        required=True,
        help="Path to a Graph .pt file or frontend .json file.",
    )
    upload_parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional Neuronpedia model id override for the result metadata.",
    )
    upload_parser.add_argument("--slug", type=str, default=None, help="Upload slug when exporting.")
    upload_parser.add_argument(
        "--graph_file_dir",
        type=str,
        default=None,
        help="Directory for intermediate frontend JSON when uploading from .pt.",
    )
    upload_parser.add_argument(
        "--node_threshold", type=float, default=DEFAULT_NODE_THRESHOLD, help="Node prune threshold."
    )
    upload_parser.add_argument(
        "--edge_threshold", type=float, default=DEFAULT_EDGE_THRESHOLD, help="Edge prune threshold."
    )
    upload_parser.add_argument(
        "-o",
        "--output",
        help="Optional path to write upload metadata JSON (default: stdout).",
    )

    args = parser.parse_args()

    if args.command == "attribute":
        run_attribution(args, attr_parser)
        if args.server:
            run_server(args)
    elif args.command == "start-server":
        run_server(args)
    elif args.command == "summarize":
        run_summarize(args)
    elif args.command == "interventions":
        run_interventions(args)
    elif args.command == "export-viz":
        run_export_viz(args)
    elif args.command == "upload-neuronpedia":
        run_upload_neuronpedia(args)


def _write_json(doc, output_path: str | None) -> None:
    import json

    payload = json.dumps(doc, indent=2)
    if output_path:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        logging.info(f"Wrote {output_path}")
    else:
        print(payload)


def run_summarize(args):
    from circuit_tracer.analysis import summarize_graph
    from circuit_tracer.graph import Graph
    from circuit_tracer.schema import validate_summary

    graph = Graph.from_pt(args.graph)
    node_threshold = None if args.no_pruning else args.node_threshold
    edge_threshold = None if args.no_pruning else args.edge_threshold
    summary = summarize_graph(
        graph,
        top_n=args.top_n,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    validate_summary(summary)
    _write_json(summary, args.output)


def run_interventions(args):
    from circuit_tracer.analysis import summarize_interventions
    from circuit_tracer.graph import Graph
    from circuit_tracer.schema import validate_summary

    graph = Graph.from_pt(args.graph)
    plan = summarize_interventions(graph, n=args.n, value=args.value)
    validate_summary(plan)
    _write_json(plan, args.output)


def run_export_viz(args):
    from circuit_tracer.utils.create_graph_files import export_graph_for_viz

    result = export_graph_for_viz(
        args.graph,
        slug=args.slug,
        output_path=args.graph_file_dir,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        neuronpedia_model=args.neuronpedia_model,
        neuronpedia_set=args.neuronpedia_set,
    )
    logging.info(result["serveCommand"])
    _write_json(result, args.output)


def run_upload_neuronpedia(args):
    from circuit_tracer.neuronpedia import upload_graph_to_neuronpedia

    result = upload_graph_to_neuronpedia(
        args.graph,
        model_id=args.model_id,
        slug=args.slug,
        output_dir=args.graph_file_dir,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )
    if result.get("url"):
        logging.info(f"Uploaded: {result['url']}")
    _write_json(result, args.output)


def run_attribution(args, parser):
    # Check if one of slug/graph_file_dir is provided but not the other
    if bool(args.slug) != bool(args.graph_file_dir):
        which_one = "slug" if args.slug else "graph_file_dir"
        missing_one = "graph_file_dir" if args.slug else "slug"
        warnings.warn(
            (
                f"You provided --{which_one} but not --{missing_one}. Both are required "
                "for creating graph files."
            ),
            UserWarning,
        )

    # Determine if we're creating graph files
    create_graph_files_enabled = args.slug is not None and args.graph_file_dir is not None

    # Validate arguments
    if args.server and (not args.slug or not args.graph_file_dir):
        parser.error("Both --slug and --graph_file_dir are required when using --server")

    if not create_graph_files_enabled and not args.graph_output_path:
        parser.error(
            "--graph_output_path is required when not creating graph files "
            "(--slug and --graph_file_dir)"
        )

    # Ensure graph output directory exists if needed
    if create_graph_files_enabled:
        assert isinstance(args.graph_file_dir, str)
        os.makedirs(args.graph_file_dir, exist_ok=True)

    import torch

    dtype = args.dtype
    # Convert short dtype string to long dtype string
    dtype_mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    if dtype in dtype_mapping:
        dtype = dtype_mapping[dtype]
    dtype = getattr(torch, dtype)

    # Run attribution
    logging.info(f"Generating attribution graph for model: {args.model}")
    logging.info(f"Loading model with dtype: {dtype}")
    logging.info(f'Input prompt: "{args.prompt}"')
    if args.graph_output_path:
        logging.info(f"Output will be saved to: {args.graph_output_path}")
    logging.info(
        f"Including logits with cumulative probability >= {args.desired_logit_prob} "
        f"(max {args.max_n_logits})"
    )
    logging.info(f"Using batch size of {args.batch_size} for backward passes")

    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    transcoder, config = load_transcoder_from_hub(
        args.transcoder_set,
        dtype=dtype,
        lazy_encoder=args.lazy_encoder,
        lazy_decoder=args.lazy_decoder,
    )
    args.model = args.model or config.get("model_name", None)
    if not args.model:
        parser.error("--model must be specified when not provided in transcoder config")

    model_instance = ReplacementModel.from_pretrained_and_transcoders(
        args.model, transcoder, dtype=dtype, backend=args.backend
    )

    logging.info("Running attribution...")
    graph = attribute(
        prompt=args.prompt,
        model=model_instance,  # type:ignore
        max_n_logits=args.max_n_logits,
        desired_logit_prob=args.desired_logit_prob,
        batch_size=args.batch_size,
        verbose=args.verbose,
        offload=args.offload,
        max_feature_nodes=args.max_feature_nodes,
    )

    # Save to file if output path specified
    if args.graph_output_path:
        logging.info(f"Saving graph to {args.graph_output_path}")
        graph.to_pt(args.graph_output_path)

    # Create graph files if both slug and graph_file_dir are provided
    if create_graph_files_enabled:
        assert isinstance(args.slug, str)
        logging.info(f"Creating graph files with slug: {args.slug}")
        create_graph_files(
            graph_or_path=graph,  # Use the graph object directly
            slug=args.slug,
            scan_name=None,  # No scan_name argument needed
            output_path=args.graph_file_dir,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )
        logging.info(f"Graph JSON files written to {args.graph_file_dir}")


def run_server(args):
    from circuit_tracer.frontend.local_server import serve

    logging.info(f"Starting server on port {args.port}...")
    logging.info(f"Serving data from: {os.path.abspath(args.graph_file_dir)}")
    if args.features_dir:
        if not os.path.isdir(args.features_dir):
            raise ValueError(f"features_dir does not exist: {args.features_dir}")
        logging.info(f"Using features directory: {os.path.abspath(args.features_dir)}")
    server = serve(data_dir=args.graph_file_dir, port=args.port, features_dir=args.features_dir)
    try:
        logging.info("Press Ctrl+C to stop the server.")
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        logging.info("Stopping server...")
        server.stop()


if __name__ == "__main__":
    main()
