# Security and Data Handling

- Keep Hugging Face and provider tokens outside Git.
- Bind local servers to `127.0.0.1` unless remote access is secured.
- Treat image content and model output as untrusted.
- Reject unknown tool names and invalid arguments.
- Validate paths before reading or writing.
- Never allow the agent to construct shell commands.
- Limit retries, response size, and HTTP timeouts.
- Do not send sensitive images to a hosted endpoint without authorization.
- Record whether inference was local or hosted.
- Preserve original media and avoid destructive overwrite.
