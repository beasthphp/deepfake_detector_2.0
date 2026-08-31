# Model runtime

This package provides the replaceable model-provider boundary used by the local API.

The runtime resolves the active model from the registry, loads the corresponding provider/artifact, normalizes model-specific outputs, and returns the stable prediction contract expected by the rest of the application.

Keeping model loading behind this boundary allows the current experimental CNN to be replaced without rewriting the browser extension or HTTP API.
