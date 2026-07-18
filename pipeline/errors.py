class JobCancelled(Exception):
    """Raised cooperatively when a web UI job's Cancel button is clicked - checked at
    stage boundaries and between long-running per-item loops (subprocess output lines,
    per-batch/per-flag LLM calls) so cancellation takes effect promptly without needing
    to forcibly kill an in-flight network call. Not used by the CLI (Ctrl+C already
    works there via KeyboardInterrupt)."""
