from cortex.inference_engine import StreamDeltaNormalizer


def test_stream_normalizer_passes_deltas():
    normalizer = StreamDeltaNormalizer()
    assert normalizer.normalize("Hello") == "Hello"
    assert normalizer.normalize(" world") == " world"


def test_stream_normalizer_handles_cumulative_chunks():
    normalizer = StreamDeltaNormalizer()
    assert normalizer.normalize("Hello") == "Hello"
    assert normalizer.normalize("Hello world") == " world"
    assert normalizer.normalize("Hello world!") == "!"


def test_stream_normalizer_handles_overlap():
    normalizer = StreamDeltaNormalizer()
    assert normalizer.normalize("abc") == "abc"
    assert normalizer.normalize("abcdef") == "def"
    # Overlap of "def" between previous total and new chunk
    assert normalizer.normalize("defghi") == "ghi"


def test_stream_normalizer_preserves_repeats_in_delta_mode():
    normalizer = StreamDeltaNormalizer()
    assert normalizer.normalize("ha") == "ha"
    assert normalizer.normalize("ha") == "ha"


def test_stream_normalizer_dedupes_long_identical_chunks():
    normalizer = StreamDeltaNormalizer()
    chunk = "Based on the provided directory listing, here is the project structure."
    assert normalizer.normalize(chunk) == chunk
    assert normalizer.normalize(chunk) == ""


def test_stream_normalizer_dedupes_in_cumulative_mode():
    normalizer = StreamDeltaNormalizer()
    assert normalizer.normalize("hello") == "hello"
    assert normalizer.normalize("hello world") == " world"
    assert normalizer.normalize("hello world") == ""
