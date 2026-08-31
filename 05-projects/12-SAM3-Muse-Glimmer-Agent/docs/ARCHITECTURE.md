# Architecture

The agent accepts a validated request, invokes one bounded segmentation tool, validates every detection, calculates aggregate area deterministically, and exports JSON plus CSV. The `demo` adapter performs measurable color segmentation. The `real` adapter loads a user-supplied SAM 3 bridge through `SAM3_PLUGIN_MODULE`; this keeps vendor-specific APIs outside the stable orchestration layer.

Trust boundaries: natural-language goals never become shell commands; tool arguments are typed; retries are bounded; backends must identify themselves; summaries use only returned measurements.
